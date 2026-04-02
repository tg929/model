# Decoder逆合成任务修复

## 背景

当前项目的 only-decoder 逆合成链路是：

1. 从 `USPTO-full/uspto_data.csv` 中读取 atom-mapped USPTO 反应。
2. 用 `USPTO-full/extract_retrosyn_data.py` 基于产物 atom map 提取真正参与成键的前体，形成 `product -> reactants`。
3. 用 `USPTO-full/prepare_only_decoder_data.py` 聚合成 only-decoder 训练样本：
   - `source_text = product>>`
   - `target_text = reactants`
4. 用 `decoder/train_retrosyn_only_decoder.py` 训练 GPT-style decoder。
5. 用 `decoder/eval_retrosyn_only_decoder.py` 做 beam-search 生成评测。

当前用户关心的问题是：

- only-decoder 在 1000 条测试上的 top-k 成功率偏低。
- 需要先判断这是模型本身的问题，还是评测和搜索实现的问题。
- 当前已知 `encoder` 权重已经具备，后续可能会走 conditioned encoder-decoder 路线，但在那之前需要先把 only-decoder 的评测判断做干净。

本文件最初用于记录诊断分析。当前第一阶段 beam-search 修复已经落地，文档同时保留：

- 修复前的分析依据
- 修复原则
- 已落地的第一阶段代码变更

## 当前项目链路梳理

### 1. 数据抽取定义

`USPTO-full/extract_retrosyn_data.py` 的核心逻辑是：

- 读取原始 `ReactionSmiles`
- 只保留单产物反应
- 用产品端 atom map 找到左侧真正贡献原子的前体分子
- 去掉 atom map
- 对 product 和 reactants 做 canonical SMILES
- 将多个前体按字典序排序后用 `.` 拼接

这一步的目标定义是清楚的：

- 输入：`product`
- 输出：`reactants`
- 不把未贡献原子的溶剂、催化剂、纯试剂保留到 target

对应代码位置：

- `mapped_precursors()` in `USPTO-full/extract_retrosyn_data.py`

### 2. only-decoder 数据准备

`USPTO-full/prepare_only_decoder_data.py` 做了几件关键事：

- 调用 `mapped_precursors()` 重新从原始映射数据抽取 `product/reactants`
- 按 `(product, reactants)` 去重聚合
- 统计 token 长度和 `[UNK]`
- 过滤过长序列和 `[UNK]`
- 做 product-exclusive split
- 生成 `train.jsonl / val.jsonl / test.jsonl`

关键点：

- 训练输入不是无条件生成，而是条件前缀生成
- `source_text = product>>`
- `target_text = reactants`

因此这里的 only-decoder 本质上是：

`P(reactants | product>>)`

对应代码位置：

- `build_training_record()` in `USPTO-full/prepare_only_decoder_data.py`

### 3. 训练目标

`decoder/train_retrosyn_only_decoder.py` 中，数据样本被编码成：

- `seq = [BOS] + source_ids + target_ids + [EOS]`
- `input_ids = seq[:-1]`
- `labels = seq[1:]`
- source 端 token 对应的 label 被改成 `pad_token_id`

也就是说：

- 模型能看到 `product>>`
- loss 只在 reactants 侧计算
- 这符合单向 decoder 条件生成的设定

所以从训练目标上看，这个 only-decoder 方案本身没有明显写错。它不是“把 product 也拿去预测”的错误训练。

对应代码位置：

- `JsonlRetrosynDataset.__getitem__()` in `decoder/train_retrosyn_only_decoder.py`

### 4. 验证和测试的差别

训练期间使用的验证指标是 teacher-forced loss：

- `val_loss`
- `val_perplexity`

测试阶段使用的是 beam search top-k exact/canonical/maxfrag。

因此训练时“最优 checkpoint”与测试时“beam top-k 最优 checkpoint”并不一定相同。这一点在当前实验里必须明确。

对应代码位置：

- `evaluate_loss()` in `decoder/train_retrosyn_only_decoder.py`
- `main()` in `decoder/eval_retrosyn_only_decoder.py`

## 当前实验事实

### 1. 数据规模

`USPTO-full/processed_only_decoder/summary.json` 记录的关键规模如下：

- `num_extracted_rows = 1,705,815`
- `num_pair_rows_before_filter = 633,559`
- `num_pair_rows = 632,053`
- `train/val/test = 505,641 / 63,206 / 63,206`
- split 之间 product overlap 为 `0`

这说明当前数据切分比传统随机 pair split 更难，因为测试产品在训练里完全不可见。

### 2. 10 epoch 训练链

当前用户重点关心的实验是：

- `decoder_runs/only_decoder_650m_10epoch`

从日志汇总看：

- `epoch1 best_val_loss = 0.14188`
- `epoch2 best_val_loss = 0.12679`
- `epoch3 best_val_loss = 0.12078`
- `epoch4_best best_val_loss = 0.11965`
- `epoch5` 之后开始回升

因此，“第四个 epoch 的验证效果最好”这件事，对 teacher-forced 验证是成立的。

### 3. 当前 1000 条评测数字

`decoder_test_results/test1000_epoch4_/test1000_best_metrics.json` 中记录：

- top-1 exact = `0.145`
- top-3 exact = `0.225`
- top-5 exact = `0.238`
- top-10 exact = `0.255`
- top-1 invalid = `0.007`

如果只看这组数字，会自然怀疑：

- 模型是不是根本没学到 product 到 reactants 的映射
- 或者 only-decoder 路线是否天然不行

但在进一步排查 beam search 后，这个结论不能直接成立。

## Beam Search 当前实现梳理

### 1. 评测入口如何调用 beam search

`decoder/eval_retrosyn_only_decoder.py` 会对每个测试样本：

1. 构造 prefix：
   - `[BOS] + tokenize(product>>)`
2. 调用：
   - `model.beam_search_generate(...)`
3. 拿到 `beam_width` 条输出
4. 解码为字符串
5. 统计 top-k exact / canonical / maxfrag

注意当前评测调用时的关键参数：

- `temperature = 0.0`
- `beam_width = 10`
- `top_k = None`
- `rp = 1.0`
- `kv_cache = False`
- `is_simulation = True`

这意味着：

- 当前结果不受 `kv_cache` 潜在 bug 影响
- 当前结果主要由 beam search 的打分规则决定

### 2. Beam Search 主流程

`decoder/model.py` 中 `beam_search_generate()` 的主流程是：

1. 初始 beam 为一个前缀序列，分数 `0.0`
2. 对每个未结束 beam：
   - 前向得到最后一个 token 的 logits
   - 转成 log-prob
   - 取 top `beam_width` 个 token 扩展
3. 把所有扩展候选按分数排序
4. 保留前 `beam_width` 个
5. 若候选都结束则停止
6. 对最终候选再做一次长度项修正
7. 返回前 `num_return_sequences` 个答案

如果只有这层骨架，那么它和任务本身是匹配的，因为它在搜索：

`argmax_y log p(y | product)`

问题在于，它并没有只搜索模型概率，而是额外加了几条硬编码规则。

## 当前 Beam Search 的两个关键问题

### 问题一：硬编码禁掉 token id 21/26/32

代码位置：

`decoder/model.py`

```python
if ((new_token == 21).sum() + (new_token == 26).sum() + (new_token == 32).sum() != 0):
    new_score = -200000
```

这不是按 token 字符串写的，而是按 token id 写死的。

对当前 vocab 来说，这三个 id 实际对应：

- `21 -> '2'`
- `26 -> '3'`
- `32 -> '4'`

也就是说：

- 只要 beam 在某一步想生成 SMILES 里的 `2/3/4`
- 这条候选就被直接打到极低分
- 后续几乎不可能留在 beam 中

### 这个规则为什么严重冲突逆合成任务

在 SMILES 里：

- `1/2/3/4/...` 是 ring closure 标记
- 简单单环结构常常只需要 `1`
- 多环、并环、桥环、复杂杂环经常必须用到 `2/3/4`

而逆合成任务的 reactants 里，这类结构大量存在。也就是说，这条规则不是“清理噪声”，而是在系统性打压合法真实前体。

### 这条规则的实证影响

对现有 `test1000_epoch4_` 预测结果做统计后，发现：

- 1000 个 target 中，有 `557` 个含 `2/3/4`
- 当前 top-1 预测中，含 `2/3/4` 的只有 `1` 个
- 1000 个样本的 top-10 候选里，只有 `1` 个样本出现过任何含 `2/3/4` 的候选
- 更关键的是：
  - 对那 `557` 个 target 含 `2/3/4` 的样本
  - 当前 top-1 / top-3 / top-5 / top-10 exact 命中全部为 `0`

而剩余 `443` 个不含 `2/3/4` 的样本上，当前结果是：

- top-1 exact = `145 / 443 = 32.7%`
- top-3 exact = `225 / 443 = 50.8%`
- top-5 exact = `238 / 443 = 53.7%`
- top-10 exact = `255 / 443 = 57.6%`

这说明当前整体 `14.5% / 25.5%` 被这个硬编码规则显著压低，不能把它当成模型真实上限。

### 问题二：固定长度惩罚

代码位置：

`decoder/model.py`

```python
candidates[i] = (candidates[i][0] - (len(frag) - len(idx)) * 0.2, frag, candidates[i][2])
```

含义是：

- 生成出来的每一个 token，最后再额外扣 `0.2`

这会形成一个很强的“偏向短序列”的后处理项。

### 这个规则为什么与任务冲突

逆合成目标不是“生成最短可能前体”，而是“生成完整、正确的前体集合”。这会自然导致：

- 多组分 reactants
- 较长的保护基或活化试剂
- 复杂骨架的真实前体

都可能比分子更短、但化学上错误的假候选吃亏。

本来 beam search 使用累计 log-prob，就已经天然偏向短序列；现在又额外每 token 扣一次固定分，会进一步压制正确但较长的答案。

从现有 1000 条预测的统计看：

- 平均 target 字符长度：`46.854`
- 平均 top-1 预测字符长度：`43.298`
- 平均 target 组分数：`1.867`
- 平均 top-1 预测组分数：`2.102`

这说明模型不是单纯“更短”，而是经常用多个较短的高频片段去拼接候选。固定长度惩罚与这种高频片段拼接倾向叠加，会进一步把搜索往“短且常见”的方向推。

## 当前评测结果应该怎么理解

### 1. 不能直接认为 only-decoder 完全失败

当前 1000 条评测里，存在明确的 beam search 污染。因此：

- `top-1 exact = 14.5%`
- `top-10 exact = 25.5%`

不能直接解释成“模型真实只能做到这个水平”。

更合理的解释是：

- 模型本身已经学到了一部分条件映射
- 但最终搜索时被硬编码规则剪掉了大量合法答案

### 2. teacher-forced 验证和 beam top-k 的脱钩依然存在

即使去掉 beam bug，也不能保证：

- teacher-forced `best_val_loss` checkpoint

一定等于

- beam-search `best_topk` checkpoint

所以后续修复后，仍然需要在小规模 val generation 上加 top-k 指标，不能只看 loss。

### 3. 数据多解性不是主因

对测试集统计后发现：

- 多 target product 在测试产品中占比不高
- 把评测放宽为“同 product 的任一 target 算对”后
- top-1 只从 `14.5%` 提升到 `16.1%`

说明多解性确实存在，但不是当前低分数的主因。主因还是 beam search 的硬编码偏置。

## 为什么这些规则像历史遗留逻辑

`beam_search_generate()` 中的两条规则：

- 禁 `2/3/4`
- 固定长度惩罚

都不像是为当前逆合成任务专门设计的。

原因是：

1. 规则直接写 raw token id，不随 vocab 语义显式绑定，可维护性很差。
2. 这类规则没有在当前评测脚本中显式参数化，也没有注释解释化学动机。
3. 文件里同一个函数还包含 `linker` 分支，说明它可能来自更早的别的生成任务，后来直接沿用到了 retrosynthesis。

因此更合理的判断是：

- 当前 beam search 混入了历史遗留的 task-specific heuristic
- 这些 heuristic 不适合当前的 retrosynthesis only-decoder 评测

## 当前修复原则

在真正动代码前，本轮分析建议遵循以下原则：

### 原则一：默认 beam 搜索应该只反映模型概率

也就是：

- 不再保留 undocumented 的 raw-id 禁词
- 不再让历史任务遗留规则污染 retrosynthesis

### 原则二：长度项必须显式化

如果保留长度项，应该：

- 参数化
- 默认关闭或设为 `0`
- 由评测脚本显式传入

而不是硬编码在 beam search 内部。

### 原则三：先得到干净评测，再判断模型路线

在 beam bug 修复前，不应该基于当前 1000 条结果就下结论说：

- only-decoder 架构一定不行
- 必须立刻放弃 only-decoder

更合理的顺序是：

1. 修 beam search
2. 重跑 `epoch4_best` 的 1000 条
3. 重跑全测试集
4. 再看 only-decoder 到底离目标差多少
5. 若仍明显不足，再讨论 encoder-decoder 封装

## 建议的后续落地顺序

### 第一阶段：修评测

优先修：

- 删除或关闭 `2/3/4` 硬编码禁词
- 把长度惩罚参数化，默认先关闭

然后重跑：

- `epoch4_best` 的 1000 条测试
- `epoch4_best` 的全测试集

目标是先得到一个“评测干净版基线”。

### 第二阶段：修验证选择标准

训练时保留 teacher-forced loss，但额外增加：

- 小规模 val generation top-k

比如：

- 每次 epoch end 在固定 500 到 1000 条 val 上计算 top-1 / top-3 / top-5 / top-10

这样后续选 best checkpoint 时，才能和真实任务目标对齐。

### 第三阶段：再决定是否引入 encoder

如果修完评测后 only-decoder 仍明显不足，再进入下一层讨论：

- 冻结 encoder
- 用 encoder 提供 product 表征
- 通过 cross-attention 或 adapter 将条件注入 decoder

这时再讨论 encoder-decoder 封装，决策会更干净，也更有依据。

## 本轮结论

本轮最重要的结论不是“only-decoder 完全不行”，而是：

1. 当前训练目标本身没有明显写错。
2. 当前 `epoch4` 作为 teacher-forced best checkpoint 是成立的。
3. 当前 1000 条测试结果被 beam search 中的硬编码规则显著污染。
4. 特别是对含 `2/3/4` ring digit 的真实前体，当前 top-k 几乎是系统性失败。
5. 因此当前的 `14.5% / 25.5%` 不能当作干净的 only-decoder 能力上限。

结论上，下一步最优先的工作不是盲目继续训练，而是：

- 先修 beam search
- 再重跑评测
- 再判断 only-decoder 还差多少

在这个基础上，再决定是否进入 encoder-decoder 集成。

## 2026-04-02 第一阶段代码修复已落地

当前已经完成第一阶段 beam-search 清理，目标是让 retrosynthesis 评测默认反映模型本身的条件概率，而不是混入历史遗留启发式。

### 已修改内容

1. `decoder/model.py`

- 删除了 beam search 中对 token id `21/26/32` 的硬编码极大惩罚。
- 这三项在当前 vocab 下对应 `2/3/4` ring digit，不应在 retrosynthesis 默认搜索中被禁掉。

2. `decoder/model.py`

- 将长度惩罚改为显式参数 `length_penalty`
- 默认值设为 `0.0`

3. `decoder/eval_retrosyn_only_decoder.py`

- 新增 `--length-penalty` CLI 参数
- 默认值为 `0.0`
- 输出 metrics 时会记录本次评测使用的 `length_penalty`

4. `decoder_runs/run_only_decoder_eval.py`

- 新增并透传 `--length-penalty`
- 便于后续对比：
  - 干净搜索基线：`0.0`
  - 若将来需要实验性长度项，也可以显式复现

### 关于旧长度惩罚实现的补充

在真正修改代码时又确认了一个实现细节：

- 原先代码里写的是 `(len(frag) - len(idx)) * 0.2`
- 但 `frag` 和 `idx` 都是形如 `[1, seq_len]` 的张量
- `len(tensor)` 取到的是第 0 维，也就是 batch 维，而不是序列长度

所以旧实现中的长度惩罚并不是按设计那样工作。它的主要问题不只是“值不合理”，而是：

- 逻辑不透明
- 含义和真实行为不一致
- 后续很难判断搜索结果到底受没受它影响

本次修复后，长度项的行为变成明确的：

- 使用 `frag.shape[1] - idx.shape[1]` 计算真实生成长度
- 再乘以显式传入的 `length_penalty`
- 默认关闭

### 当前阶段结论

现在的 beam search 默认行为更接近：

`argmax_y log p(y | product)`

而不是：

`argmax_y log p(y | product) + 历史遗留禁词 + 隐式长度偏置`

因此，下一步重新跑 `epoch4_best` 的 1000 条和全测试集时，得到的结果才适合作为 only-decoder 的干净基线。

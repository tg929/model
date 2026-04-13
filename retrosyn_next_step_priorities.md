# 逆合成下一步优先级与清洗清单

状态: `living document`

日期: `2026-04-08`

## 1. 用途

这份文档不是一次性总结，而是接下来围绕 only-decoder retrosynthesis 继续讨论时的主记录文档。用途有三件事:

1. 固定当前已经形成共识的优先级
2. 把需要人工回查的数据清洗名单落成可执行清单
3. 在后续讨论中持续同步新的决定、假设和行动项

参考分析:
- `decoder_test_results/testall_epoch4_beamfix/analysis_report.md`
- `decoder_test_results/testall_epoch4_beamfix/testall_best_metrics.json`
- `decoder_test_results/testall_epoch4_beamfix/testall_best_predictions.jsonl`
- `USPTO-full/processed_only_decoder/test.jsonl`

## 2. 当前共识的优先级

当前推荐顺序:

1. 先做 `reranker`
2. 再做 `数据清洗`
3. 最后再决定是否继续堆 `epoch`

这不是泛泛建议，而是由当前全量评估结果直接支持的。

## 3. 为什么 `reranker` 是第一优先级

当前全量测试集 `63,206` 条样本上的关键结果:

| 指标 | 数值 |
| --- | ---: |
| top-1 exact | `0.4007` |
| top-3 exact | `0.5456` |
| top-10 exact | `0.6291` |

换成更直接的计数:

- top-1 exact 命中: `25,327`
- top-10 exact 命中: `39,766`
- beam 额外带来的 exact 命中: `14,439`

这些 beam 带来的额外命中里:

- `rank 2-3`: `9,160`
- `rank 4-5`: `2,898`
- `rank 6-10`: `2,381`

结论:
- 正确答案大量已经在候选集合里
- 当前损失很大一部分不是“模型不会生成”，而是“正确答案没有排在第 1”
- 所以先做 `candidate reranking / rescoring` 的性价比最高

### 3.1 这一步为什么比继续加 epoch 更优先

如果正确答案根本不在 beam 候选里，继续训练可能是第一优先级。

但当前不是这个状态:
- `top10 exact - top1 exact = 0.2284`
- 说明排序误差已经足够大，值得单独处理

因此在没有先解决排序问题前，继续堆 epoch 容易出现两种低回报情况:

- 训练继续强化高频近邻试剂混淆，但 top-1 排序改进有限
- teacher-forced 验证继续改善，但 generation-time top-1 不同步改善

### 3.2 当前建议的最小 reranker 目标

第一阶段不需要马上做复杂系统，先回答一个明确问题:

> 对同一个 beam 候选集合，只通过重排，能把 full-test top-1 exact 从 `0.4007` 拉到多少?

建议第一阶段只做:
- 固定现有生成模型和 beam 配置
- 对每个样本的 top-k 候选做重排
- 用 full-test top-1 exact 作为主指标

第一阶段成功标准:
- top-1 exact 明显上升
- top-10 exact 基本不变
- invalid rate 不上升

### 3.3 当前工作方案: `v1 reranker` 先不加额外特征

当前建议把第一版 reranker 控制为一个干净的基线实验:

- 只对现有 beam 候选做重排
- 不改变生成模型
- 不引入新的 learned feature model
- 不把人工化学先验和数据清洗规则先混进 reranker 里

原因:
- 现在最需要先回答的是“单靠更合理的候选打分，能回收多少 top-1”
- 如果 v1 一上来就混入额外特征，后面很难判断收益到底来自:
  - 更合理的序列打分
  - 还是手工规则 / 特征偏置
- 当前测试集还存在可疑标签噪声，过早加入额外特征容易把数据问题和排序问题混在一起

因此建议 v1 目标定义为:

> 在固定 beam 候选集合上，只做纯重排，先测出 top-1 exact 的干净增益。

当前更合适的 v1 形式:
- 用相同 decoder 对 `product -> candidate reactants` 做 teacher-forced 条件打分
- 只统计 target-side token 的条件 log-likelihood
- 对长度做归一化，避免 beam 累积分数天然偏向短序列

v1 暂不建议混入:
- 组件数惩罚
- 可疑分子惩罚
- 手工模板分
- 外部分类器特征

这些内容更适合作为 v2，在 v1 基线跑通之后再加。

### 3.4 为什么主分数用 `target-side mean log-prob`

定义:

对于给定产品 `x` 和候选反应物序列 `y = (y_1, ..., y_T)`，

- `sum log-prob = Σ_t log p(y_t | x, y_<t)`
- `mean log-prob = (1 / T) * Σ_t log p(y_t | x, y_<t)`

其中 `T` 只统计 target-side token，可包含 `EOS`，不包含 source prefix。

把 `mean log-prob` 设成主分数，核心原因有三个。

第一，它和当前训练目标最一致。

当前 only-decoder 训练在 `decoder/train_retrosyn_only_decoder.py` 里:
- 把 `source_text` 和 `target_text` 拼成一个序列
- 但把 source 部分的 label 全部 mask 掉
- 最终 loss 只在 target-side token 上计算

因此模型真正被训练去最小化的，本质上就是:
- target-side token 的平均负对数似然

换句话说:
- `mean log-prob` 最接近模型训练时真正优化的对象
- 它不是额外发明出来的分数，而是训练目标在 reranking 阶段的自然延续

第二，它能主动消掉长度偏置。

`sum log-prob` 有一个天然问题:
- 序列越长，累积 log-prob 通常越小
- 即使一个长候选在每个 token 上都更合理，它也可能因为 token 更多而总分更差

这在当前任务里尤其麻烦，因为:
- reactants 长度差异很大
- 组件数差异很大
- `1` 组分和 `3+` 组分样本的 target 长度不是一个量级

如果直接把 `sum log-prob` 当主分数，reranker 很容易学成:
- 更偏爱短候选
- 更偏爱缺组分的候选
- 更偏爱“看起来像半句正确答案”的候选

而 `mean log-prob` 的作用就是:
- 把“总分”转成“平均每个 token 的置信度”
- 让长短不同的候选至少先站在更公平的比较基线之上

第三，它更容易解释。

`mean log-prob` 反映的是:
- 给定产品后，模型对这个候选 reactants 序列的平均 token-level 置信度

它更接近:
- 每个 token 平均有多意外
- 整条候选序列在模型眼里平均有多顺

这和 `per-token perplexity` 是同一类量，只是一个取对数、一个取指数。

### 3.5 为什么把 `target-side sum log-prob` 留作对照分数

虽然 `sum log-prob` 不适合作为 v1 主分数，但它仍然必须保留，原因也很明确。

第一，它是语言模型最原始的联合概率分数。

`sum log-prob` 直接对应:
- 模型对整条候选序列 `y` 的条件联合概率 `P(y | x)`

从概率论角度，它最“正统”。

第二，它能作为长度偏置的对照组。

如果之后看到:
- `mean log-prob` 明显优于 `sum log-prob`

那基本可以直接说明:
- 当前排序问题里，长度偏置是重要因素

反过来如果:
- `sum log-prob` 更好

那说明当前模型在 beam 候选里可能更需要:
- 对完整序列总证据的偏好
- 而不是强长度归一化

也就是说，把 `sum` 留作对照，不是因为它更简单，而是因为它能帮助判断:
- reranking 的收益到底来自“去掉长度偏置”
- 还是来自“重新用 teacher-forced 条件概率评估候选”

### 3.6 这两个分数各自反映什么

`mean log-prob` 主要反映:
- 候选序列逐 token 的平均条件置信度
- 模型觉得这条候选在局部生成上是否自然、顺滑、一致

`sum log-prob` 主要反映:
- 模型愿意为整条候选支付多少总概率质量
- 一个候选作为“完整序列整体”在模型里的总证据强弱

两者的区别可以简化成一句话:

- `mean` 更像“平均质量”
- `sum` 更像“总证据”

### 3.7 为什么 v1 不加组件数惩罚

因为组件数本身现在就是问题的一部分，不应该在 v1 里先手工规定答案。

如果一上来加组件数惩罚:
- 你等于把“我们认为多少组分才合理”的先验写进 reranker
- 这会掩盖模型本身到底会不会区分正确候选

而且当前测试集还存在标签边界噪声:
- 有些样本可能本来就多塞了工艺性分子
- 这时组件数惩罚反而会把数据问题和排序问题搅在一起

所以 v1 应该先回答:
- 只靠模型自己的条件概率，能不能把正确候选排得更靠前

### 3.8 为什么 v1 不加可疑分子惩罚

因为这会把 reranking 和 data cleaning 混成一件事。

一旦你对 `THF / Et3N / DMF` 之类先加惩罚:
- reranker 提升了，也很难知道是排序真的更强
- 还是只是人为压低了那批本来就可疑的候选

当前更合理的流程是:

1. reranker 先保持纯概率打分
2. 数据清洗单独审计
3. 后面再讨论是否把 clean-set 结论变成规则型特征

### 3.9 为什么 v1 不加外部化学特征

因为 v1 的任务不是追求最强系统，而是先建立一个干净、可归因的基线。

如果现在就加入:
- 反应模板特征
- 分子图特征
- 组件数特征
- 人工规则特征

会立刻带来三个问题:

1. 收益来源难以解释
2. 实验空间迅速膨胀
3. 一旦 full-test 指标变化，不知道是排序改进还是额外特征在补数据噪声

所以 v1 更像一个“诊断实验”:
- 它要先回答纯 decoder 概率是否足以显著改善 top-1
- 如果答案是可以，再决定 v2 是否值得引入更复杂特征

### 3.10 `mean log-prob` 里为什么倾向于把 `EOS` 也计入

当前倾向是:

- `v1` 的 `mean log-prob` 计算里应包含 `EOS`
- 也就是把完整候选看成 `target tokens + EOS`

更精确地写就是:

- 不含 `EOS`:
  - `score_no_eos = (1 / T) * Σ_t log p(y_t | x, y_<t)`
- 含 `EOS`:
  - `score_with_eos = (1 / (T + 1)) * [Σ_t log p(y_t | x, y_<t) + log p(EOS | x, y)]`

当前更倾向 `score_with_eos`，原因有四个。

第一，当前训练目标本来就包含 `EOS`。

在 `decoder/train_retrosyn_only_decoder.py` 中，训练序列构造是:
- `BOS + source + target + EOS`

虽然 source 位置被 mask 掉了，不参与 loss，但:
- target token 参与 loss
- 末尾 `EOS` 也参与 loss

所以模型学到的并不是“怎样写出一个 target prefix”，而是:
- “怎样写出完整的 target，并在正确位置结束”

第二，从概率定义上说，包含 `EOS` 才是在打分完整序列事件。

不含 `EOS` 时，你评分的是:
- `P(y | x)` 的前缀部分

含 `EOS` 时，你评分的是:
- `P(y, EOS | x)`

这两者不一样。

前者更像是在问:
- “这串 reactant token 看起来像不像一个合理前缀?”

后者才是在问:
- “模型是否认为这是一条完整、应当在这里结束的答案?”

第三，`EOS` 本身就是长度与完整性的信号。

如果不把 `EOS` 计入:
- 候选是否该在这里停住不会被显式奖励或惩罚
- 一个局部 token 都很顺，但本质上该继续写下去的候选，可能被高估
- 一个少了最后一段的小候选，可能因为更短反而更占便宜

把 `EOS` 计进去之后，分数会同时反映:
- 候选内容是否合理
- 候选是否在正确位置自然结束

第四，当前 retrosynthesis 评估路径本来就是靠 `EOS` 结束。

在当前 beam-search eval 中，`is_simulation=True` 时实际终止符就是 `EOS`，不是 `SEP`。
因此从实现角度看，`EOS` 不是可有可无的附属符号，而是当前任务里真实的停止决策。

### 3.11 `EOS` 计入分数后，分数多反映了什么

如果把 `EOS` 也算进去，`mean log-prob` 反映的不再只是:
- 平均每个 reactant token 有多可信

而是同时反映:
- 这串 reactants 本身是否可信
- 模型是否认为它已经完整，可以在这里结束

因此它更接近:
- “完整答案平均有多可信”

而不是:
- “答案正文这个前缀平均有多可信”

### 3.12 为什么这对当前任务是好事

因为 retrosynthesis 里一个常见错误正是:
- 组分没写全
- 组分多写了一截
- 主体对了，但完整答案边界不对

把 `EOS` 计进去后，reranker 不只是判断“正文像不像”，还会判断:
- “这个候选是不是正好该停在这里”

这与当前任务需求更一致。

### 3.13 当前需要注意的实现边界

虽然当前倾向于把 `EOS` 计入，但真正落实现时要统一两件事:

1. 对每个候选都要显式补上 `EOS` 再做 teacher-forced 打分
2. 如果某个 beam 候选本身是截断产物，而不是自然结束样本，要定义一致规则，避免把“没结束”候选和“已结束”候选混着算

当前工作建议是:
- `v1` 先把 `EOS` 计入
- 另保留一个 `no-EOS` 对照分数，作为 ablation

这样后面如果看到差异，就能明确知道:
- 终止位置判断本身，对 reranking 到底贡献了多少

### 3.14 当前锁定的 `v1 reranker` 打分方案

基于当前讨论，先把 `v1 reranker` 的打分方案锁定如下。

主分数:
- `mean(target + EOS)`

第一对照:
- `mean(target)`

第二对照:
- `sum(target + EOS)`

这里的含义统一为:
- `target`: beam 候选中的 reactants token 序列
- `EOS`: 在候选末尾显式补上的停止 token
- `mean`: 除以参与打分的 target token 数量；`target + EOS` 的 mean 则除以 `T + 1`
- `sum`: 不做长度归一化

这套设计的意图非常明确:

- 用 `mean(target + EOS)` 作为主分数，度量“完整答案平均有多可信”
- 用 `mean(target)` 判断主收益里到底有多少来自正文 token 本身
- 用 `sum(target + EOS)` 判断去掉长度偏置到底有多重要

因此当前不再把以下选项作为 `v1` 待决项:
- 是否把 `EOS` 计入主分数
- 是否把 `sum(target)` 作为主对照
- 是否在 `v1` 里额外加入规则惩罚项

这些都先固定，避免 `v1` 范围继续扩散。

### 3.15 `mean(target + EOS)` 的具体实现规则

为了避免实现阶段再出现歧义，`v1` 主分数的计算规则进一步固定如下。

#### 3.15.1 一律显式补 `EOS`

对每个 beam 候选，不论它在生成时是否自然以 `EOS` 结束，reranker 计算主分数时都统一按下面的目标序列处理:

- `target_ids = tokenizer.encode(candidate_text, add_special_tokens=False)`
- `scored_target = target_ids + [eos_token_id]`

也就是说:
- reranker 不依赖预测文件里是否保存了原始 `EOS`
- 而是对候选文本统一补一个 `EOS` 再做 teacher-forced 条件打分

这样做的意义是:
- 让所有候选都在同一评分语义下比较
- 即都被解释成“如果这条 candidate 就是完整答案，那么模型是否愿意在这里结束”

#### 3.15.2 截断候选不做特殊奖励，只靠 `EOS` 项自然惩罚

这里区分两类“截断”:

1. `generation-truncated`
   - beam 候选因为到达生成长度上限而停止
   - 并不是模型主动生成了 `EOS`
2. `scoring-overflow`
   - scorer 自身因为实现或显存限制，无法完整打分这条候选

当前建议:

- 对 `generation-truncated` 候选:
  - 不做额外手工惩罚
  - 直接按 `candidate_text + EOS` 计算主分数
  - 如果该候选本来不该在这里结束，`log p(EOS | prefix)` 会自然偏低，从而自动拉低分数

- 对 `scoring-overflow` 候选:
  - 不允许静默截断后继续打分
  - 要么完整打分
  - 要么记录为 `scoring_failed` 并排到最后

原因:
- 静默裁掉尾部再算 `mean`，会人为抬高短候选和截断候选
- 这会破坏 `v1` 要保持干净可解释的原则

#### 3.15.3 分母统一按“被打分 token 数”计算

主分数 `mean(target + EOS)` 的分母固定为:

- `len(target_ids) + 1`

也就是:
- 只统计 target token
- 再加上那个显式补上的 `EOS`

明确不使用以下分母:
- 不除以 source 长度
- 不除以整条拼接序列长度
- 不按字符数除
- 不按组件数除
- 不按最大长度上限除

这样分母反映的是:
- 实际被评分的目标 token 数量

#### 3.15.4 建议的实现方式

对产品前缀 `source_text` 与候选 `candidate_text`，建议按如下逻辑计算:

1. `source_ids = tokenizer.encode(source_text, add_special_tokens=False)`
2. `target_ids = tokenizer.encode(candidate_text, add_special_tokens=False)`
3. 构造:
   - `seq = [BOS] + source_ids + target_ids + [EOS]`
4. 前向时只取 target-side 与 `EOS` 对应位置的 log-prob
5. 再分别计算:
   - `mean(target + EOS)`
   - `mean(target)`
   - `sum(target + EOS)`

这样实现会与当前训练脚本的 masking 语义保持一致。

### 3.16 `v1 reranker` 输入输出文件格式

当前正式锁定如下。

#### 路径

`v1 reranker` 相关文件统一放在:

- `decoder_test_results/testall_epoch4_beamfix/reranker_v1/`

原因:
- 这一版 reranker 明确建立在 `testall_epoch4_beamfix` 的 beam 候选之上
- 把 reranker 产物和原始评估结果放在同一个结果目录下，最利于追踪 provenance

#### 输入文件格式: `JSONL`

输入文件定为:

- `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_input.jsonl`

格式定为“每个样本一行”的自包含 `JSONL`，而不是一行一个 candidate。

建议 schema:

```json
{
  "sample_idx": 0,
  "first_id": "US04788214;;183391",
  "first_year": 1988,
  "product": "B(C1CCCCC1)C1CCCCC1",
  "source_text": "B(C1CCCCC1)C1CCCCC1>>",
  "target_text": "B.C1=CCCCC1",
  "canonical_target": "B.C1=CCCCC1",
  "beam_width": 10,
  "candidates": [
    {
      "rank": 1,
      "text": "CC(C)OB(C1CCCCC1)C1CCCCC1",
      "canonical_text": "CC(C)OB(C1CCCCC1)C1CCCCC1",
      "maxfrag_text": "CC(C)OB(C1CCCCC1)C1CCCCC1"
    }
  ]
}
```

为什么用 sample-level `JSONL`:
- 与现有 `predictions.jsonl` 风格一致
- 一个样本天然对应一组 beam candidates
- 候选数是数组结构，`JSONL` 比 `CSV` 更适合
- 自包含后，reranker 不必再同时依赖原 predictions 文件和 test JSONL 做 join

#### 输出明细格式: `JSONL`

输出明细定为:

- `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_scored.jsonl`

同样使用 sample-level `JSONL`，在输入记录基础上追加评分与重排结果。

建议 schema:

```json
{
  "sample_idx": 0,
  "product": "...",
  "source_text": "...",
  "target_text": "...",
  "candidates": [...],
  "scores": {
    "mean_target_eos": [-0.42, -0.51, -0.63],
    "mean_target": [-0.39, -0.48, -0.55],
    "sum_target_eos": [-11.8, -10.1, -16.4]
  },
  "reranked_candidate_indices": {
    "mean_target_eos": [2, 0, 1],
    "mean_target": [2, 0, 1],
    "sum_target_eos": [1, 0, 2]
  },
  "top1_before": "...",
  "top1_after_mean_target_eos": "..."
}
```

为什么输出也用 `JSONL`:
- 评分结果是“每个样本对应多个 candidate score”
- 这是嵌套结构，不适合直接压成单表 `CSV`
- 之后做离线分析时，可以继续按样本逐行流式处理

#### 指标汇总格式: `JSON`

汇总文件定为:

- `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_metrics.json`

内容至少包括:
- 输入文件路径
- 三套分数名称
- 主分数名称
- before / after 的 top-k exact、canonical、maxfrag
- invalid rate
- reranker 失败样本数

为什么汇总用 `JSON`:
- 与当前 eval metrics 文件保持一致
- 汇总天然是一个单对象，不需要 `JSONL`

#### 可选调试输出

如果需要进一步人工检查，可额外写:

- `decoder_test_results/testall_epoch4_beamfix/reranker_v1/v1_reranker_errors.jsonl`

只保留:
- `before hit`
- `after hit`
- 分数翻转失败样本
- 以及特殊边界样本

但这个文件不是 `v1` 必需主产物。

## 4. 为什么 `数据清洗` 是第二优先级

当前测试集中存在一批很像溶剂、碱或工艺性分子的目标组分，它们与当前项目定义的任务边界存在明显张力。

当前项目定义是:
- 目标应该是 `true reactants`
- 不应把不贡献原子的 solvent / catalyst / reagent-only molecules 放进 target

但测试集里能看到一批疑似违反这一边界的组分，例如:
- `Et3N`
- `THF`
- `DMF`
- `diethyl ether`
- `ethyl acetate`
- `ethanol`
- `methanol`

对这批样本做了一个 review shortlist 统计后发现:

- 按这批分子的并集去看，相关样本约 `2,665` 条，占测试集约 `4.22%`
- 这批样本的 exact-match 表现显著差于其余样本

之前的对比结果:

| 样本类型 | top-1 exact | top-10 exact |
| --- | ---: | ---: |
| 含可疑工艺分子 | `0.0949` | `0.2432` |
| 其余样本 | `0.4142` | `0.6461` |

结论:
- 这部分不应简单看成“模型太弱”
- 更合理的解释是: 这里混入了值得回查的目标边界噪声

注意:
- 这里的“可疑”不等于“一定错误”
- 有些小分子在个别反应里确实可能是原子贡献前体
- 所以这一步叫 `review / audit`，不是直接粗暴删除

## 5. 为什么 `继续堆 epoch` 应该放到第三优先级

当前瓶颈已经不只是“训练不够久”。

至少有三件事情比继续训练更先暴露出来:

1. 排序误差很大
2. `3+` 组分复杂样本性能断崖下降
3. 测试集与目标定义之间可能存在一部分边界噪声

在这三个问题没有先澄清前，继续堆 epoch 的信息价值有限。

更具体地说:
- 如果 reranker 就能显著抬高 top-1，说明主问题不是继续训练
- 如果清洗后指标显著上升，说明一部分“错误”其实是标签噪声
- 只有在这两步之后，继续训练的收益才更容易被正确解释

## 6. 可疑标签分子清单

下面这份表不是“删除名单”，而是“优先人工回查名单”。

| 分子 | 常见含义 | 样本数 | 占测试集 | top-1 exact | top-10 exact | 建议优先级 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `CCN(CC)CC` | Et3N / triethylamine | `556` | `0.88%` | `0.0306` | `0.1655` | `P0` |
| `C1CCOC1` | THF | `425` | `0.67%` | `0.0024` | `0.0447` | `P0` |
| `CN(C)C=O` | DMF | `445` | `0.70%` | `0.1236` | `0.3551` | `P1` |
| `CCN(C(C)C)C(C)C` | DIPEA / Huenig base | `216` | `0.34%` | `0.0324` | `0.1343` | `P0` |
| `CCOCC` | diethyl ether | `167` | `0.26%` | `0.0000` | `0.0719` | `P0` |
| `CCOC(C)=O` | ethyl acetate | `194` | `0.31%` | `0.0103` | `0.0722` | `P0` |
| `CCO` | ethanol | `304` | `0.48%` | `0.1546` | `0.3224` | `P1` |
| `CO` | methanol | `486` | `0.77%` | `0.2551` | `0.4712` | `P2` |

优先级解释:
- `P0`: 极像工艺性组分，且命中率极低，应优先抽样审查
- `P1`: 可疑度高，但个别反应里确有可能参与，建议第二批回查
- `P2`: 需要谨慎，不能默认是噪声，更适合人工确认后再决定规则

## 7. 示例样本 ID

下面这些 `first_id` 适合作为第一轮人工审查入口。

### `CCN(CC)CC` / Et3N

- `US07767651B2;0178;1000405`
- `US07410991B2;0269;850446`
- `US05612289;;363954`

### `C1CCOC1` / THF

- `US07732365B2;0147;986213`
- `US07589076B2;0236;924293`
- `US07074819B2;0766;733672`

### `CN(C)C=O` / DMF

- `US07642367B2;0458;949078`
- `US09273067B2;0501;1733541`
- `US06936602B1;1215;695520`

### `CCN(C(C)C)C(C)C` / DIPEA

- `US20150376130A1;0124;1807998`
- `US20150291597A1;0635;1768494`
- `US08906889B2;0661;1549941`

### `CCOCC` / diethyl ether

- `US20140083752A1;0038;1504765`
- `US04247549;;77459`
- `US04254036;;78809`

### `CCOC(C)=O` / ethyl acetate

- `US05612289;;363954`
- `US06967256B2;0162;704834`
- `US20110070192A1;0250;1028461`

### `CCO` / ethanol

- `US07687490B2;0501;964470`
- `US04563449;;141170`
- `US04886835;;203945`

### `CO` / methanol

- `US20100166765A1;0658;904790`
- `US06001849;;459945`
- `US07605261B2;0468;932980`

## 8. 第一轮清洗讨论建议

为了避免一上来把问题做散，建议按下面顺序讨论:

### 8.1 第一轮只讨论 `P0`

优先看:
- `Et3N`
- `THF`
- `DIPEA`
- `diethyl ether`
- `ethyl acetate`

目标不是立刻删，而是先回答:

1. 这些分子在当前 target 中出现时，有多少其实是工艺性组分?
2. 它们是由 atom mapping 决策失真带来的，还是 extraction 规则本身就需要收紧?
3. 是应该做“规则修复”，还是只做“审计标签后生成 clean split”?

### 8.3 当前工作方案: `THF + Et3N` 先审

当前建议先从:

- `THF` (`C1CCOC1`)
- `Et3N` (`CCN(CC)CC`)

开始，而不是一上来把全部 `P0` 混在一起看。

原因:
- 两者都高度可疑，且样本量足够大
- `THF` 更代表典型溶剂类问题
- `Et3N` 更代表典型有机碱类问题
- 如果这两类都出现大比例“目标里混入非原子贡献分子”，就足以支持后续回头修 extraction 边界

当前建议的最小审计切片:

1. `THF` top-1 错误样本抽样 `50`
2. `Et3N` top-1 错误样本抽样 `50`
3. `THF` 命中样本再抽样 `20`
4. `Et3N` 命中样本再抽样 `20`

这样做的目的不是只看“错样本有多糟”，还要看:
- 命中的那部分里，这两个分子是否真的有化学贡献
- 是否存在少量真实贡献案例，导致后续规则不能一刀切

第一轮审计输出应该至少回答:

1. `THF` 在 target 中出现时，明显属于工艺性组分的比例有多高?
2. `Et3N` 在 target 中出现时，明显属于工艺性组分的比例有多高?
3. 这些问题主要来自 atom-map 继承、left/middle 合并策略，还是 extraction 判断过宽?
4. 后续更应该做:
   - 抽取规则修复
   - clean subset
   - 还是两者都做

### 8.4 `THF / Et3N` 审计模板

第一轮人工审计建议不要直接在自由文本里做，而是落成结构化表。

建议最小字段如下:

| 字段 | 说明 |
| --- | --- |
| `focus_molecule` | 当前审计对象，取 `THF` 或 `Et3N` |
| `sample_bucket` | `top1_hit` / `top1_miss_top10_hit` / `top10_miss` |
| `first_id` | 当前样本的审计主键入口 |
| `first_year` | 年份，辅助看老数据偏差 |
| `product` | 产品 SMILES |
| `target_text` | 当前标签 reactants |
| `top1_prediction` | 当前模型 top-1 预测 |
| `focus_molecule_judgment` | `true_contributor` / `non_contributing_process_molecule` / `ambiguous` |
| `target_action` | `keep_as_is` / `remove_focus_molecule` / `exclude_row` / `unclear` |
| `root_cause_hypothesis` | `mapping_leak` / `role_merge_issue` / `rule_too_permissive` / `model_error` / `unclear` |
| `notes` | 人工备注 |

这里最重要的不是把字段做多，而是保证后续能直接把审计结果转成:
- clean subset
- root cause 汇总

#### 8.4.1 为什么要有 `sample_bucket`

因为我们不是只想看“标签对不对”，还想知道错误是如何影响模型行为的。

`sample_bucket` 至少分三类:

- `top1_hit`
  - 标签与模型完全一致
  - 用来检查“命中的样本里，这个分子是否仍然是可疑噪声”
- `top1_miss_top10_hit`
  - 模型候选里已经包含正确标签，但没排第一
  - 更偏 reranking 问题
- `top10_miss`
  - 模型候选里完全没覆盖到标签
  - 更偏数据边界或生成难度问题

这样后续分析时，才能区分:
- 是标签噪声导致模型“学不会”
- 还是标签没问题，但排序有问题

### 8.5 修正后的抽样方案

之前口头上提到“每类各抽 50 个错误样本 + 20 个命中样本”，但这个方案对 `THF` 不成立。

当前统计下:
- `Et3N`:
  - top-1 hits 约 `17`
  - top-10 hits 约 `92`
- `THF`:
  - top-1 hits 约 `1`
  - top-10 hits 约 `19`

所以第一轮应改成“按 bucket 自适应抽样”，而不是固定命中数。

建议方案:

#### `THF`

- `top1_hit`: 全审
- `top1_miss_top10_hit`: 全审
- `top10_miss`: 再抽 `40`

这会得到一个小而完整的 high-suspicion 切片。

#### `Et3N`

- `top1_hit`: 全审
- `top1_miss_top10_hit`: 抽 `20`
- `top10_miss`: 抽 `30`

这样能同时覆盖:
- 少量正例
- 排序问题
- 完全未覆盖问题

抽样原则:
- 固定随机种子
- 尽量按 `year bucket` 与 `count bucket` 做简单分层
- 避免样本全部集中在某几年或某一类高频专利模板

### 8.6 审计结果如何回写

当前建议的顺序不是“先改抽取规则”，而是:

1. 先把人工审计结果写成结构化审计表
2. 再基于审计表生成一个 `clean subset`
3. 等 root cause 模式足够稳定后，再决定是否回头修 extraction 规则

原因:
- clean subset 能最快告诉我们“去掉明显边界噪声后，指标能回升多少”
- 而 rule fix 是更慢、更重、影响更广的动作

因此第一轮审计输出后，建议先支持两种回写方式:

#### 回写方式 A: clean subset

基于 `target_action` 生成:
- `keep_as_is`
- `remove_focus_molecule`
- `exclude_row`

然后形成一个人工审计版本的评估子集，用于:
- 重新算指标
- 估计标签噪声的上界影响

#### 回写方式 B: rule-fix backlog

基于 `root_cause_hypothesis` 统计:
- `mapping_leak`
- `role_merge_issue`
- `rule_too_permissive`

只有当某一类模式足够集中时，才把它升级成 extraction 规则修复任务。

### 8.7 `THF / Et3N` 审计文件路径与格式

当前正式锁定如下。

#### 路径

审计相关文件统一放在:

- `decoder_test_results/testall_epoch4_beamfix/audits/`

第一轮审计对象固定为:

- `decoder_test_results/testall_epoch4_beamfix/audits/thf_et3n_round1_cases.jsonl`
- `decoder_test_results/testall_epoch4_beamfix/audits/thf_et3n_round1_audit.csv`

#### 为什么不是只放一个文件

因为审计有两类信息:

1. 机器生成的样本上下文
2. 人工填写的判定结果

这两类信息最适合分开存。

#### 样本清单: `JSONL`

样本清单文件定为:

- `thf_et3n_round1_cases.jsonl`

作用:
- 固定这一轮被抽中的样本集合
- 保留样本原始上下文
- 作为人工审计的只读输入

建议字段:
- `focus_molecule`
- `sample_bucket`
- `sample_idx`
- `first_id`
- `first_year`
- `product`
- `target_text`
- `top1_prediction`
- `top10_predictions`

为什么这里用 `JSONL`:
- `top10_predictions` 是数组
- 后续如果要补更多上下文字段也不容易破坏格式

#### 人工审计表: `CSV`

人工填写文件定为:

- `thf_et3n_round1_audit.csv`

为什么这里明确用 `CSV` 而不是 `JSONL`:
- 人工审计表本质上是一个扁平表格
- `CSV` 最适合直接用电子表格工具打开、筛选、批注
- 后续汇总 `judgment`、`target_action`、`root_cause_hypothesis` 时也最方便

建议列:
- `focus_molecule`
- `sample_bucket`
- `sample_idx`
- `first_id`
- `first_year`
- `product`
- `target_text`
- `top1_prediction`
- `focus_molecule_judgment`
- `target_action`
- `root_cause_hypothesis`
- `notes`

当前建议:
- `cases.jsonl` 只读
- `audit.csv` 人工填写

这样能避免把机器上下文和人工修改混在同一个文件里。

### 8.2 第二轮再讨论 `P1/P2`

`DMF`、`ethanol`、`methanol` 需要更谨慎，因为:
- 它们更容易在个别反应里既像工艺分子，也可能真参与
- 不适合一刀切删除

## 9. 后续讨论要持续同步的事项

这部分是接下来每次讨论后都要更新的“同步区”。

### 9.1 当前已确认

- 当前 full-test 的第一优先级是 `reranker`
- 当前 full-test 的第二优先级是 `可疑标签审计`
- 在这两步之前，不建议把“继续堆 epoch”当成默认主路线
- 当前工作方案里，`v1 reranker` 先做纯候选重排，不先混入额外特征
- 当前工作方案里，`P0` 审计从 `THF + Et3N` 开始
- 当前已锁定的 `v1 reranker` score set 为:
  - 主分数: `mean(target + EOS)`
  - 第一对照: `mean(target)`
  - 第二对照: `sum(target + EOS)`
- `mean(target + EOS)` 的实现规则已固定为:
  - 对所有候选统一显式补 `EOS`
  - generation-truncated 候选不加手工惩罚，交给 `EOS` 概率自然处理
  - scoring-overflow 候选不得静默截断，失败则排到最后
  - 分母固定为 `len(target_ids) + 1`
- `THF / Et3N` 第一轮审计模板已固定为结构化字段表，不再采用自由文本随手记录
- `THF / Et3N` 第一轮抽样改为按 `top1_hit / top1_miss_top10_hit / top10_miss` 三个 bucket 自适应抽样
- `v1 reranker` 的主产物路径与格式已固定为:
  - sample-level input `JSONL`
  - sample-level scored output `JSONL`
  - single-object metrics `JSON`
  - all under `decoder_test_results/testall_epoch4_beamfix/reranker_v1/`
- `THF / Et3N` 审计文件布局已固定为:
  - read-only sampled cases `JSONL`
  - human annotation table `CSV`
  - all under `decoder_test_results/testall_epoch4_beamfix/audits/`
- `v1 reranker` full-run 已完成，当前结果为:
  - 原始 beam 顺序: `top1 exact=0.4007`, `top10 exact=0.6291`
  - `mean(target + EOS)`: `top1 exact=0.3947`, `top10 exact=0.6291`
  - `mean(target)`: `top1 exact=0.3888`, `top10 exact=0.6291`
  - `sum(target + EOS)`: `top1 exact=0.4007`, `top10 exact=0.6291`
- 当前已确认:
  - 这版纯 teacher-forced 候选重排没有带来 full-test `top1 exact` 提升
  - `sum(target + EOS)` 基本等价于原 beam 排序
  - 长度归一化的 `mean` 系列在当前 full-test 上反而略差
  - `THF / Et3N` 的下一步执行工具已经齐备:
    - enriched audit context: `decoder_test_results/testall_epoch4_beamfix/audits/thf_et3n_round1_context.jsonl`
    - audit-driven clean-subset outputs: `thf_et3n_round1_effective_subset.jsonl` 与 `thf_et3n_round1_clean_subset_metrics.json`
  - `thf_et3n_round1_audit.csv` 已完成首轮全量填写（`126/126`）
  - 全量 `126` 条结果一致指向:
    - `non_contributing_process_molecule`
    - `remove_focus_molecule`
    - `mapping_leak`
  - 当前审计覆盖进度:
    - `total=126`, `included=126`, `pending=0`
    - `THF`: `59/59` 已审
    - `Et3N`: `67/67` 已审
  - 审计后的 clean-subset 指标（baseline beam order）:
    - 对原始 target: `top1 exact=0.1429`, `top10 exact=0.4444`
    - 对 effective target: `top1 exact=0.0159`, `top10 exact=0.1349`
  - 已落地的抽取规则修复策略（`audit_v1_fix`）:
    - 位置: `USPTO-full/extract_retrosyn_data.py`
    - 开关: `--apply-audit-v1-fix`
    - 规则: 先按 map overlap 选前体，再移除不在 demapped product 中的 `THF/Et3N` (`C1CCOC1`, `CCN(CC)CC`)
    - 可扩展: `--process-molecule-smiles` 追加 blocklist
  - `prepare_only_decoder_data.py` 已支持同一策略开关，并在 `summary.json` 记录:
    - `apply_audit_v1_fix`
    - `process_molecule_blocklist`
  - clean-subset 报告口径已固化为可重放脚本:
    - 脚本: `decoder_runs/render_clean_subset_report.py`
    - 产物: `decoder_test_results/testall_epoch4_beamfix/audits/thf_et3n_round1_clean_subset_report.md`
  - 长任务的增量落盘能力已补齐（默认每 `1000` 条）:
    - `USPTO-full/extract_retrosyn_data.py`: `--progress-every`, `--progress-json`
    - `USPTO-full/prepare_only_decoder_data.py`: `--progress-every`, `--progress-json`
    - `decoder_runs/render_clean_subset_report.py`: `--progress-every`, `--progress-json`
  - 口径差异提示:
    - 现有按行 `focus` 的 effective-target 定义，与 `audit_v1_fix` 全局 blocklist 定义在 `3` 条含 `THF+Et3N` 的样本上会不同
    - 后续汇报需明确声明使用的是哪一种 effective-target 口径
  - 这轮全量结果进一步强化当前判断:
    - `THF / Et3N` 样本里 process-molecule mapping leakage 非孤例，而是系统性问题
    - 当前模型在这类样本上经常把泄漏分子一并预测出来，导致 effective target 下 exact 明显下降

### 9.2 当前待决问题

1. 在 `v1` 纯重排失败之后，下一步应优先转向 `THF / Et3N` 数据审计，还是继续做带 score fusion / 额外特征的 `v2 reranker`?
2. 数据清洗应该走“修抽取规则”还是“基于现有 split 再生成 clean subset”?
3. 后续主汇报指标应该用:
   - 原始 full-test
   - clean subset
   - 还是两者同时保留
4. `3+` 组分复杂样本是否应单独建立一套分析和实验节奏?

### 9.3 文档同步规则

从这份文档建立之后，后续每次形成新结论时，至少同步以下两处:

1. 更新本文件 `retrosyn_next_step_priorities.md`
2. 更新 `memory-bank/progress.md`

只有在文件角色、执行路径、模块边界变化时，才额外更新:
- `memory-bank/architecture.md`

## 10. 当前的执行建议

基于当前 `v1 reranker` full-run 结果和 `THF / Et3N` 首轮全量审计结果，建议顺序收缩为:

1. 把审计结论转成可复现实验口径（明确 effective-target 评估口径与报告口径）
2. 决定数据清洗路径：先做 split 后 clean-subset，还是回到抽取规则侧修复
3. 在统一后的口径上再判断是否值得做 `v2 reranker`
4. 最后再决定是否继续追加训练轮数

原因很直接:

- 当前这版最干净的 pure reranker baseline 已经跑完，且没有把 full-test `top1 exact` 从 `0.4007` 往上拉
- `THF / Et3N` 首轮审计 `126/126` 已完成，并确认了强烈的标签边界泄漏信号
- 所以后续优先级应转向“先固定数据与评估口径”，再讨论更复杂 reranker

一句话总结:

> 当前阶段最值得投入的，不是继续打磨同一版 pure reranker，而是先把“标签边界问题”审清，再决定是否需要更强的 `v2 reranker`。

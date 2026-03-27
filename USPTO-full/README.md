# USPTO-full

This folder is reserved for downloading and preparing the original USPTO reaction data with `reaction-utils`.

## Preparation

Activate an environment that contains `rxnutils`, then run:

```bash
bash prepare_uspto_full.sh
```

The script runs:

```bash
python -m rxnutils.data.uspto.preparation_pipeline run --folder . --nbatches 200 --max-workers 8 --max-num-splits 200
```

You can override the defaults with environment variables:

```bash
NBATCHES=200 MAX_WORKERS=8 MAX_NUM_SPLITS=200 bash prepare_uspto_full.sh
```

## Atom Mapping

After preparation, run:

```bash
bash map_uspto_full.sh
```

The script runs:

```bash
python -m rxnutils.data.mapping_pipeline run --data-prefix uspto --nbatches 200 --max-workers 8 --max-num-splits 200
```

You can override the defaults in the same way:

```bash
DATA_PREFIX=uspto NBATCHES=200 MAX_WORKERS=8 MAX_NUM_SPLITS=200 bash map_uspto_full.sh
```

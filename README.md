# Structured Dendrite

This repository is a compact experiment framework for dendrite+soma sequence models.

The active entrypoint is `train.py`, the active Python package is `structured_dendrite/`, and the active Hydra config tree is `conf/`.

## What This Repo Is Optimized For

The main research axis here is the dendrite.

Instead of keeping one near-duplicate model file per variant, the code uses one backbone with interchangeable dendrite modules:

- `identity`: soma-only ablation.
- `s4d`: structured state-space dendrite used inside the spiking dendrite+soma block.
- `conv1d`: temporal convolution dendrite.
- gla: gated linear attention dendrite.
- pointwise_mlp: static per-token control with no temporal mixing.
- `s4d_standard`: a non-spiking debug baseline meant to be closer to a standard S4D block for reproduction checks.

The default spiking block is:

`dendrite -> soma (LIF) -> spike -> output projection`

The new `s4d_standard` debug mode instead uses a residual sequence block without the soma/spike path, so it is easier to compare against the official S4 setup.

## Environment Setup

A good default on the local Windows dev machine and on a typical remote Linux x86_64 server is a Python 3.11 conda environment.

```powershell
conda create -n structured-dendrite python=3.11 -y
conda activate structured-dendrite
python -m pip install --upgrade pip
pip install -e .
```

If you want the optional GLA research dependency too:

```powershell
pip install -e .[research]
```

`pip install -e .` now pulls the exact CUDA-enabled PyTorch 2.10.0 / torchvision 0.25.0 `cu130` wheels from the official PyTorch host on these tested combinations:

- Windows `AMD64` + Python 3.11
- Linux `x86_64` + Python 3.11

If your remote server uses a different Python version, a different architecture, or a different preferred CUDA build, install the matching PyTorch wheel from the official PyTorch instructions first, then run `pip install -e .`.

A practical Linux server flow is:

```bash
conda create -n structured-dendrite python=3.11 -y
conda activate structured-dendrite
python -m pip install --upgrade pip
pip install -e .
```

### Making Codex Use The Environment

For me to run commands inside the correct environment, use one of these workflows:

1. Launch Codex from a terminal where `conda activate structured-dendrite` is already active.
2. Tell me the environment name and ask me to prefix commands with `conda run -n structured-dendrite ...`.
3. If you already created a different env name, tell me that exact name and I can use it.

In practice, option 2 is the most reliable if Codex was not launched from the activated shell.

### Quick Smoke Test Commands

First verify the dependencies import:

```powershell
conda run -n structured-dendrite python -c "import torch, lightning, datasets, transformers; print('ok')"
```

Then run a lightweight training smoke test on a Hugging Face task that does not require local files:

```powershell
conda run -n structured-dendrite python train.py experiment=s4_debug/imdb trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1 data.loader.num_workers=0 run.test_after_fit=false
```

A WT103 smoke test is heavier, but a reduced version is:

```powershell
conda run -n structured-dendrite python train.py experiment=s4_debug/wt103 trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1 data.loader.num_workers=0 data.max_length=128 data.lm_stride=128 data.loader.batch_size=1 data.loader.eval_batch_size=1 run.test_after_fit=false
```

## Active Structure

- `train.py`: Hydra + Lightning entrypoint.
- `conf/`: active Hydra config tree.
- `structured_dendrite/data/`: data loading, tokenization, and collation.
- `structured_dendrite/models/backbone.py`: shared encoder/backbone/decoder logic.
- `structured_dendrite/models/dendrites/`: S4D, Conv1D, GLA, identity, and optimizer metadata helpers.
- `structured_dendrite/models/spiking/`: truncated LIF soma and surrogate spike function.
- `structured_dendrite/experiment.py`: Lightning module, optimizer, scheduler, logging, and trainer wiring.

## Running

Typical usage:

```powershell
python train.py experiment=liq_ssm/listops_rs trainer.devices=[0]
python train.py experiment=appendix/listops_ssm_upper_bound trainer.devices=[0,1]
python train.py experiment=liq_ssm/wt103 trainer.devices=[0]
python train.py experiment=s4_debug/listops trainer.devices=[0]
```

`experiment=...` is usually the main thing you change. Most experiment files are thin presets that choose a task, a dataset, a model family, and sometimes a few task-specific overrides.

## How Hydra Composition Works Here

The top-level config is `conf/config.yaml`. It selects one file from each config group:

- `paths`
- `task`
- `data`
- `model`
- `optimizer`
- `scheduler`
- `trainer`
- `logging`
- `experiment`

Hydra merges them into one runtime config.

A useful mental model is:

- `conf/data/*.yaml` says what the dataset looks like and how to tokenize/collate it.
- `conf/model/*.yaml` says what network block to build.
- `conf/task/*.yaml` says how to decode outputs and what metric to monitor.
- `conf/experiment/*.yaml` chooses a coherent combination and optionally overrides a few values.

## The `conf/` Folder In Detail

### `conf/config.yaml`

This is the root composition file.

It sets the default stack to:

- `task=classification`
- `data=listops`
- `model=ssm_reservoir`
- `optimizer=adamw`
- `scheduler=cosine`
- `trainer=default`
- `logging=csv`
- `experiment=liq_ssm/listops_rs`

It also defines:

- `run.name`: experiment name prefix.
- `run.seed`: global seed.
- `run.resume_from`: optional checkpoint path.
- `run.test_after_fit`: whether to run test after training.
- `run.test_only`: skip fit and run test only.
- `hydra.run.dir`: output directory pattern.

### `conf/data/`

This group describes dataset structure and preprocessing. Every file in `conf/data/` follows the same schema.

Common fields:

- `task_name`: semantic label for the task. Currently `classification` or `language_modeling`.
- `input_kind`: which data path to use. Choices here are `text`, `pair_text`, `image_sequence`, and `language_model`.
- `num_classes`: number of target classes for classification tasks.
- `max_length`: default token length for classification, or LM block size for language modeling.
- `train_max_length`: optional train-only token/block length override.
- `eval_max_length`: optional validation/test token/block length override.
- `train_fraction`: optional train split subsampling ratio for low-data studies.
- `max_train_examples`: optional cap on train examples after shuffling.
- `max_eval_examples`: optional cap on validation/test examples for smoke runs.
- `lm_stride`: stride used when chopping LM text into blocks. `null` means not used.
- `label_field`: dataset column containing labels.
- `loader.batch_size`: training batch size.
- `loader.eval_batch_size`: validation/test batch size. Falls back to `batch_size` if null.
- `loader.num_workers`: PyTorch DataLoader workers.
- `splits.*`: names of train/validation/test splits in the underlying dataset.
- `source.path`: what `datasets.load_dataset()` should load.
- `source.*`: extra arguments passed to Hugging Face `load_dataset()`.
- `text.primary_field`: main text column.
- `text.secondary_field`: second text column for pair tasks such as AAN.
- `image.field`: image column name for image datasets.
- `image.channels`: expected image channels after conversion.
- `tokenizer.type`: tokenization mode. Supported local modes are `char`, `whitespace`, and `wordpunct`. The code also supports `huggingface` if you set `name_or_path`.
- `tokenizer.name_or_path`: Hugging Face tokenizer id when `tokenizer.type=huggingface`.
- `tokenizer.min_frequency`: minimum token frequency when building a local vocabulary tokenizer.
- `tokenizer.lowercase`: whether to lowercase before tokenization.
- `tokenizer.add_bos`: prepend beginning-of-sequence token.
- `tokenizer.add_eos`: append end-of-sequence token.

Dataset-specific notes:

- `listops.yaml`: local TSV classification task. Expects `text` and `label` columns.
- `aan.yaml`: local TSV pair-text task. Expects `text1`, `text2`, and `label` columns.
- `imdb.yaml`: Hugging Face IMDB classification task.
- `cifar.yaml`: Hugging Face CIFAR-10, converted from images into sequences of pixels.
- `pathfinder.yaml`: local `imagefolder` dataset rooted at `${paths.data_root}/pathfinder`.
- `wt103.yaml`: Hugging Face WikiText-103 raw text, chunked into fixed-length next-token prediction blocks.

Local-vs-remote behavior:

- `imdb`, `cifar10`, and `Salesforce/wikitext` download through Hugging Face.
- `listops`, `aan`, and `pathfinder` expect local files under `${paths.data_root}`.
- `paths.data_root` defaults to `./data` and can be changed with `DATA_ROOT=/your/path`.

### `conf/model/`

This group defines the backbone family and its internal hyperparameters.

Common top-level fields:

- `d_model`: hidden width of the model.
- `n_layers`: number of backbone layers.
- `dropout`: dropout used inside blocks.
- `prenorm`: whether each block uses pre-layernorm instead of post-layernorm.
- `final_norm`: optional final layernorm on the backbone output. Used mainly by the `s4d_standard` residual baseline.
- `max_positions`: maximum length available if learned positional embeddings are enabled.
- `tie_embeddings`: tie LM output weights to token embedding weights.
- `block.mode`: backbone block type. `spiking` is the dendrite+soma path, `residual` is the non-spiking debug baseline.

`model.soma.*` fields:

- `soma.threshold`: spike threshold used by the surrogate spiking nonlinearity.
- `soma.truncation_steps`: how many recurrent steps the truncated LIF approximation keeps.
- `soma.optim.lr`, `soma.optim.weight_decay`: special optimizer settings for the LIF parameters `rou`, `a`, and `b`.

`model.dendrite.*` fields:

- `kind`: dendrite family. `identity`, `pointwise_mlp`, `s4d`, `conv1d`, `gla`, or `s4d_standard`.
- `direction`: `causal` or `bidir`. Bidirectional runs both forward and reverse processing and combines them. In this repo the bidirectional path should be treated as an optional noncausal upper bound rather than the main biologically motivated setting.
- `freeze_all`: freezes all dendritic parameters after initialization. This is the main setting for the reservoir/fixed-dendrite ablation.
- `freeze_dynamics`: mainly meaningful for S4D. If true, the continuous-time dynamics stay fixed while other dendritic parameters may still train.
- `freeze_processor`: freezes the main processor weights for Conv1D, GLA, pointwise controls, or the SSM readout-like terms.
- `freeze_skip`: freezes additive skip/input-scale parameters inside the dendrite.
- `freeze_processor`: especially relevant for Conv1D, GLA, pointwise controls, and the S4D readout-like processor terms.
- `d_state`: S4D state size.
- `dt_min`, `dt_max`: S4D initialization range for discretization timescales.
- `input_scale_init`: initialization for the additive skip/input-scale parameter.
- `kernel_size`: Conv1D kernel size.
- `use_branch_mixer`: whether Conv1D adds an optional 1x1 branch-mixing stage after depthwise causal filtering.
- `branch_mixer_groups`: grouping for the optional 1x1 branch mixer.
- `mlp_hidden_multiplier`: hidden width multiplier for the pointwise MLP control.
- `gla_gate_normalizer`: gate normalization constant used by the official-style GLA parameterization.
- `groups`: Conv1D grouping. `-1` means depthwise by setting groups to `d_model`.
- `bias`: Conv1D bias flag.
- `n_heads`: GLA attention head count.

`model.dendrite.optim.*` fields:

These define optional special optimizer groups.

- `dynamics_lr`, `dynamics_weight_decay`: special settings for sensitive SSM dynamics parameters such as `log_dt`, `log_a_real`, and `a_imag`.
- `processor_lr`, `processor_weight_decay`: special settings for processor weights such as Conv1D kernels, GLA projections, and the standard S4D output projection.
- `skip_lr`, `skip_weight_decay`: special settings for additive skip parameters such as `input_scale` or the standard S4D `D` parameter.

If a value is `null`, the global optimizer setting is used instead.

Model files in this repo:

- `soma.yaml`: dendrite identity ablation. Useful to isolate the soma/spike mechanism.
- `ssm_reservoir.yaml`: S4D dendrite with all dendritic parameters frozen after initialization. This is the main fixed-dendrite / reservoir ablation.
- `ssm_causal.yaml`: trainable causal S4D dendrite.
- `ssm_bidir.yaml`: trainable bidirectional S4D dendrite used as an optional noncausal upper bound.
- `conv1d_*`: Conv1D dendrite variants with depthwise causal filtering and an optional 1x1 branch mixer.
- `gla_*`: GLA dendrite variants.
- `s4d_standard_bidir.yaml`: non-spiking S4D baseline for LRA-style tasks.
- `s4d_standard_causal.yaml`: non-spiking S4D baseline for LM/WT103-style tasks.

### `conf/experiment/`

These files are intentionally thin. Their main job is to select a coherent `task + data + model` combination and optionally override a few values such as width, depth, LR, or warmup.

Current folders:

- `liq_ssm/`: main spiking SSM runs across tasks.
- `dend_ablation_parameter/`: main complexity story comparing soma-only, fully frozen SSM dendrites, and trainable causal SSM dendrites.
- dend_ablation_structure/: compares soma, SSM, Conv1D, and GLA structures.
- 
eviewer_controls/: static pointwise controls that test whether gains come from temporal structure rather than extra preprocessing.
- 
eviewer_claims/: low-data, short-budget, and length-generalization presets for stronger inductive-bias checks.
- ppendix/: optional noncausal upper-bound runs such as bidirectional SSM.
- `s4_debug/`: standard non-spiking S4D baselines for sanity-checking the full training setup against the official S4 family.

Typical pattern inside an experiment file:

```yaml
defaults:
  - override /task: classification
  - override /data: listops
  - override /model: ssm_reservoir
  - _self_

run:
  name: liq_ssm_listops_rs
```

That means the experiment is mostly selecting presets from other groups, not redefining everything from scratch.

### `conf/task/`

This group describes the output head and the monitored metric.

`classification.yaml`:

- `decoder_mode: pool`: pool the sequence into one representation and classify it.
- `pooling`: sequence pooling rule. Currently `mean`, `first`, or `last` are supported in code.
- `pair_mode: concat_abs_prod`: for pair tasks, use `[a, b, |a-b|, a*b]` before the classifier.
- `head_dropout`: classifier head dropout.
- `head_norm`: optional classifier head layernorm.
- `monitor_metric`: checkpoint selection metric.
- `monitor_mode`: whether larger or smaller is better.

`language_modeling.yaml`:

- `decoder_mode: sequence`: predict one token per sequence position.
- `embedding_dropout`: dropout on token embeddings.
- `output_norm`: whether to apply a layernorm before the LM head.
- `monitor_metric`: checkpoint selection metric.
- `monitor_mode`: whether larger or smaller is better.

### `conf/optimizer/`

Currently this repo uses AdamW:

- `lr`: default learning rate.
- `weight_decay`: default weight decay.
- `betas`: AdamW beta coefficients.

Special dendrite and soma parameter groups override these defaults only when explicitly configured in `model.dendrite.optim` or `model.soma.optim`.

### `conf/scheduler/`

- `name`: `cosine` or `none`.
- `warmup_steps`: linear warmup steps.
- `total_steps`: total scheduler steps. If `0`, Lightning's estimated step count is used.
- `min_lr_ratio`: final LR as a fraction of base LR for cosine decay.

### `conf/trainer/`

These map directly to Lightning trainer arguments used by `structured_dendrite/experiment.py`:

- `accelerator`: `auto`, `cpu`, `gpu`, etc.
- `devices`: device list or count.
- `max_epochs`: max training epochs.
- `gradient_clip_val`: gradient clipping value.
- `accumulate_grad_batches`: gradient accumulation steps.
- `precision`: Lightning precision mode.
- `log_every_n_steps`: logging frequency.
- `benchmark`: enable cuDNN benchmark mode.
- `deterministic`: enable deterministic execution.

### `conf/logging/`

- `kind`: `csv` or `tensorboard`.
- `save_dir`: logger output directory.
- `name`: logger run name.
- `rich_model_summary`: show model summary in terminal.
- `rich_progress_bar`: show rich progress bar.
- `checkpoint.save_top_k`: number of best checkpoints to keep.

### `conf/paths/`

- `data_root`: local dataset root. Defaults to `./data`.
- `output_root`: Hydra output root. Defaults to `./outputs`.

## Data Philosophy

The loader layer is intentionally much smaller than the inherited S4 framework.

- Text and pair-text tasks use a generic Hugging Face dataset path plus a lightweight tokenizer layer.
- Image tasks use Hugging Face image datasets or `imagefolder` sources and convert images into sequences in the collator.
- WikiText-103 currently uses the same shared tokenizer/data pipeline rather than a custom LM iterator stack.

That makes the project easier to maintain, but it also matters for result interpretation.

## Important WT103 Caveat

The current WT103 setup is now architecturally closer to a standard S4 starting point, but the data pipeline is still not an exact match to the official S4 repository.

What is aligned now:

- WT103 uses a causal non-spiking `s4d_standard` debug baseline if you choose `experiment=s4_debug/wt103`.
- The LM head is sequence-to-sequence next-token prediction.
- Sensitive S4D dynamics can use their own optimizer hyperparameters.

What is still different:

- This repo loads raw WikiText-103 text through Hugging Face.
- It tokenizes with the repo's generic tokenizer settings from `conf/data/wt103.yaml`.
- It chops the stream into fixed blocks using `max_length` and `lm_stride`.
- It uses a standard tied linear LM head.

Why that matters:

Language-model results depend heavily on tokenization, vocabulary construction, sequence packing, and the exact loss/evaluation pipeline. Even if the backbone architecture is good, changing the text pipeline can move perplexity a lot.

So the WT103 caveat is not "the model is broken." The caveat is:

- the model can train correctly,
- but a gap to official S4 WT103 numbers may come from the data/tokenization/evaluation pipeline rather than the backbone itself.

In other words, for LRA tasks the new `s4_debug/*` presets are a pretty reasonable architecture sanity check. For WT103 they are a useful LM baseline, but not yet a fully apples-to-apples reproduction of the official S4 WT103 setup.

## Practical Recommendations

If your immediate goal is sanity checking the implementation:

- Use `experiment=s4_debug/listops` or `experiment=s4_debug/imdb` first.
- Then compare `experiment=s4_debug/wt103` against `experiment=liq_ssm/wt103`.
- If WT103 is still your main target, the next improvement should be aligning the WT103 data/tokenization/evaluation pipeline more closely with the official S4 repo.

## Notes About Local Data

The configs used for the main LRA runs are aligned with the local layouts used by the official S4 repository.

Expected local layouts:

- `data/listops/basic_train.tsv`, `basic_val.tsv`, `basic_test.tsv` with `Source` and `Target` columns.
- `data/aan/new_aan_pairs.train.tsv`, `new_aan_pairs.eval.tsv`, `new_aan_pairs.test.tsv` as the headerless 5-column LRA export (`label`, `input1_id`, `input2_id`, `text1`, `text2`).
- `data/pathfinder/curv_contour_length_14/...` in the original metadata layout used by LRA. The repo now also accepts `data/pathfinder/pathfinder32/...` if the 32x32 release is nested one directory deeper, and it builds the train/validation/test split deterministically from metadata.
- `data/imdb/aclImdb/...` is used as a local fallback for the LRA Text task when Hugging Face is unavailable on the server.

The current `liq_ssm/*` presets assume the S4-style ListOps, AAN, and Pathfinder layouts above, and `pathfinder_rs` specifically targets the 32x32 release with the `curv_contour_length_14` subset.

## Package Baseline

The project currently depends on:

- `lightning`
- `torch`
- `torchvision`
- `datasets`
- `transformers`
- `hydra-core`
- `omegaconf`
- `einops`
- `torchmetrics`
- optional `flash-linear-attention`







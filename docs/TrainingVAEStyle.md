# Fine-tuning the VAE Planner with Driving Style

This document covers the style fine-tune of ORION's **original VAE planner** in `StyleOrion/`:
how the `driving_style` labels reach the planner, the config, the precision changes that let
the 7.5 B model train on one 24 GB GPU, and the launch script.

The labels themselves come from `scripts/label_bench2drive.sh`; see [DataPreparation.md](DataPreparation.md).

| File | Role |
|---|---|
| `scripts/train_vae_style.sh` | launch script (run from the repo root) |
| `StyleOrion/adzoo/orion/configs/orion_stage2_vae_style_ft.py` | the config |
| `StyleOrion/adzoo/orion/run_vae_style_resume.sh` | the launcher of the original run, kept as a record; its `CFG` and `WORK_DIR` are paths on that machine |
| `StyleOrion/mmcv/datasets/b2d_orion_dataset.py` | reads `driving_style` from the pkl |
| `StyleOrion/mmcv/models/detectors/orion.py` | style injection into the VAE planner, planner-only training |

## 1. Prerequisites

**Environment.** The ORION environment, set up as in [README.md](../README.md#orion-environment)
(Python 3.8, torch 2.4.1 + cu118, `pip install -v -e .` inside `StyleOrion/`). Note the
flash-attn 1.x import in `mmcv/models/utils/attention.py` (see [§5](#5-precision-changes)).

**Checkpoints**, under `StyleOrion/ckpts/`:

| Path | Source |
|---|---|
| `ckpts/pretrain_qformer/` | [exiawsh/pretrain_qformer](https://huggingface.co/exiawsh/pretrain_qformer/) |
| `ckpts/Orion.pth` | the official ORION checkpoint, the starting point of the fine-tune |

**Data**, under `data/Bench2Drive/`:

| Path | Written by |
|---|---|
| `raw/v1/<Scene>/`, `maps/` | `scripts/download_bench2drive.sh` |
| `infos/b2d_infos_train.pkl`, `infos/b2d_map_infos.pkl` | `style_labeling.bench2drive.infos.build_infos` |
| `infos/b2d_infos_train_with_style.pkl` | `scripts/label_bench2drive.sh` |

The config reads `data/infos/b2d_infos_train_style_mixed.pkl`. That name comes from the
original fine-tune runs, where it stood for "style-annotated frames plus neutral frames".
`build_style_pkl.py` writes every frame (labeled frames get their score, the rest `0.0`), so
`b2d_infos_train_with_style.pkl` is that file. The script links one to the other.

## 2. Running

```bash
bash scripts/train_vae_style.sh
```

The script:

1. links `StyleOrion/data/bench2drive/{v1,maps}` and `StyleOrion/data/infos` to
   `data/Bench2Drive/` (the config resolves `data/...` relative to `StyleOrion/`), and
   `b2d_infos_train_style_mixed.pkl` to `b2d_infos_train_with_style.pkl`. Existing files or
   links are left alone;
2. checks that the pkls, `pretrain_qformer/` and the start checkpoint exist;
3. launches `adzoo/orion/train.py` the same way as the original launcher: one process through
   `torch.distributed.launch`, `--launcher pytorch --deterministic`, output teed to
   `<WORK_DIR>/logs/train.<MMDDHHMM>`. It adds `--no-validate` (see below) and passes
   `load_from`, `resume_from` and `runner.max_iters` through `--cfg-options`.

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | the GPU to use (the script is single-GPU) |
| `WORK_DIR` | `StyleOrion/adzoo/orion/work_dirs/orion_stage2_vae_style_ft` | logs, dumped config and checkpoints (`iter_*.pth`, every 2000 iterations, last 3 kept) |
| `LOAD_FROM` | `StyleOrion/ckpts/Orion.pth` | weights to start from |
| `RESUME_FROM` | empty | checkpoint to resume from, with optimizer state and iteration count; overrides `LOAD_FROM` |
| `MAX_ITERS` | `20896` | total iterations; a resumed run continues up to this number |
| `CFG` | `adzoo/orion/configs/orion_stage2_vae_style_ft.py` | config, relative to `StyleOrion/` |
| `MASTER_PORT` | `54643` | change it when two runs share a machine |

Resume a stopped run:

```bash
RESUME_FROM=StyleOrion/adzoo/orion/work_dirs/orion_stage2_vae_style_ft/iter_4000.pth \
    bash scripts/train_vae_style.sh
```

`train.py` has its own `--resume-from`, but it is ignored silently when the file does not
exist, so the script passes `resume_from` through `--cfg-options` and checks the file first.

**`--no-validate`.** Without it, `train.py` builds a validation set from
`data/infos/b2d_infos_val.pkl`, which this repo does not produce. The config never evaluates
anyway, because `evaluation.interval` (40000) is larger than `max_iters`.

**Choosing `MAX_ITERS`.** With one GPU and `batch_size=1`, one iteration is one frame.
`len(pkl)` iterations is one pass over the training set:

```bash
python -c "import pickle; print(len(pickle.load(open('data/Bench2Drive/infos/b2d_infos_train_with_style.pkl','rb'))))"
```

The 36 scenes in `labeled_scenes.json` give 8384 frames.

### The original run

The config and the launcher record a two-phase run:

1. **4000 iterations from `ckpts/Orion.pth`.** The config still carries `ft_max_iters = 4000`
   and `evaluation.interval = 40000` (= `ft_max_iters * 10`), the values of the frozen
   fine-tune configs it was expanded from. The checkpoint was
   `.../orion_stage2_vae_style_ft_frozen/iter_4000.pth`.
2. **Resumed from `iter_4000` to `max_iters=20896`**, with `run_vae_style_resume.sh`. The
   banners describe the extra 16,896 iterations as one pass over a 16,896-frame training set.
   That pkl is not on this machine, so the frame count could not be checked here.

To reproduce the schedule with this script:

```bash
MAX_ITERS=4000 bash scripts/train_vae_style.sh
RESUME_FROM=StyleOrion/adzoo/orion/work_dirs/orion_stage2_vae_style_ft/iter_4000.pth \
MAX_ITERS=$((4000 + <frames in your pkl>)) bash scripts/train_vae_style.sh
```

## 3. How the style reaches the VAE planner

```
pkl 'driving_style'  (float in [-1, 1]; -1 conservative, 0 normal, +1 aggressive)
  └─ B2DOrionDataset.get_data_info()                 mmcv/datasets/b2d_orion_dataset.py
     FORCE_DRIVING_STYLE=<s> overrides every frame with a constant
       └─ CustomCollect3D keys include 'driving_style'  (train and test pipelines in the config)
          └─ Orion.forward_pts_train()                 mmcv/models/detectors/orion.py
             current_states = ego_feature + style_token_proj(style)
               ├─ CVAE prior / posterior   (present_distribution, future_distribution)
               ├─ GRU rollout              (predict_model; hidden state starts from current_states)
               └─ ego_fut_decoder          -> 6-step ego trajectory
```

- **`style_token_proj`** is `Linear(1, 4096) → Mish → Linear(4096, 4096)`. The last layer's
  weight and bias start at zero, so at iteration 0 the model behaves exactly like the VAE
  planner it was loaded from.
- **Injection by addition.** The style embedding is added to `current_states`, the LLM's ego
  feature. The prior, posterior, rollout and decoder all read it. The VAE branch always adds,
  whatever `style_inject_mode` is set to.
- **Style dropout.** During training each sample's style is zeroed with probability
  `style_dropout` (0.15), so the model also learns the unconditioned prediction. This is what
  makes classifier-free guidance possible at inference.
- **Inference** (`simple_test_pts`). The conditioned and unconditioned decodes share one latent
  noise sample. With `STYLE_GUIDANCE=g` (default `1.0`, off) the output is
  `pred_uncond + g * (pred_cond - pred_uncond)`.
- **Planner-only training.** With `train_only_planner=True` only parameters whose names start
  with `present_distribution`, `future_distribution`, `predict_model`, `ego_fut_decoder` or
  `style_token_proj` stay trainable. The vision backbone, detection and map heads and the LLM
  (with its LoRA) are frozen, and the optimizer constructor skips frozen parameters. The log
  shows `[StyleDrive] train_only_planner: trainable=...M frozen=...M` at start-up.

## 4. The config

`orion_stage2_vae_style_ft.py` is a fully expanded config (no `_base_`). The settings that
matter here:

| Key | Value | Notes |
|---|---|---|
| `data.train.ann_file` | `data/infos/b2d_infos_train_style_mixed.pkl` | |
| `CustomCollect3D.keys` | include `'driving_style'` | without it the model sees style `0` |
| `model.use_diff_decoder` | `False` | VAE planner |
| `model.use_style_conditioning` | `True` | |
| `model.style_inject_mode` | `'add'` | ignored by the VAE branch |
| `model.style_dropout` | `0.15` | |
| `model.train_only_planner` | `True` | |
| `model.use_lora` | `True` | frozen here, since it is outside the planner |
| `optimizer` | AdamW, `lr=2e-4`, `weight_decay=1e-4`, ViT-wise lr decay | |
| `optimizer_config` | grad clip, `max_norm=35` | |
| `lr_config` | cosine annealing, 200-iteration linear warm-up from `lr/3`, `min_lr_ratio=0.01` | |
| `runner` | `IterBasedRunner`, `max_iters=20896` | the script overrides it with `MAX_ITERS` |
| `checkpoint_config` | `interval=2000`, `max_keep_ckpts=3` | saved to the work dir |
| `load_from` | `<StyleOrion>/ckpts/Orion.pth` | built from the config's own location |
| `resume_from` | `None` | |
| `half_precision_load` | `True` | bf16, see [§5](#5-precision-changes) |
| `num_gpus`, `batch_size` | `1`, `1` | |
| `evaluation.interval` | `40000` | larger than `max_iters`: no evaluation |

Two settings were changed for this repo. The original values are kept as comments in the
config:

- `resume_from` was `/mnt/SSD7/college_student/outputs/orion_stage2_vae_style_ft_frozen/iter_4000.pth`
  and is now `None`.
- `checkpoint_config.out_dir` was `/mnt/HDD6/yinxuan/dow904_ckpt` and has been removed, so
  checkpoints go to the work dir.

Other settings to know about:

- `llm_path`, `tokenizer` and `lm_head` are the relative path `ckpts/pretrain_qformer/`, so
  training must run from `StyleOrion/`. The script `cd`s there.
- `num_iters_per_epoch = 234769` and the sampler's `num_iters_to_seq=234769` are the full
  Bench2Drive frame count, hard-coded upstream. `IterBasedRunner` stops at `max_iters`, so
  neither sets the length of this run.

## 5. Precision changes

The model has about 7.5 B parameters. In fp32 its weights alone take about 36 GB and do not
fit on a 24 GB RTX 3090 / 4090. Training runs in **bf16**, switched on by
`half_precision_load=True`. bf16 has fp32's exponent range, so no loss scaler is needed. The
original fp16 path overflowed when training this model.

Each change is marked in its file with a `[Bench2Drive-Style] MODIFIED FILE -- precision tuning`
banner.

**Loading and training in bf16**

| File | Change |
|---|---|
| `mmcv/utils/fp16_utils.py` | `auto_fp16` casts to `torch.bfloat16` and opens `autocast(dtype=bfloat16)` instead of fp16 |
| `adzoo/orion/apis/mmdet_train.py` | with `half_precision_load`, `model.to(torch.bfloat16)` **before** `.cuda()`, so the fp32 weights are never on the GPU. `runner.resume(..., map_location='cpu')`: the checkpoint with optimizer state is about 18 GB and is staged in CPU RAM. `cfg.static_graph` calls `DDP._set_static_graph()` (not set by this config) |
| `adzoo/orion/test.py` | the same bf16 cast before `.cuda()` for offline testing |

**Ops without a bf16 kernel**

| File | Change |
|---|---|
| `mmcv/models/utils/distributions.py` | the GRU in `PredictModel` runs with cuDNN disabled and contiguous inputs. cuDNN's RNN rejects bf16. **This is the VAE planner's rollout**, so it matters directly here |
| `mmcv/ops/focal_loss.py` | the sigmoid focal-loss CUDA kernel has no bf16 version: inputs are cast to fp32 for the call and the output back to bf16 |

**Mixed dtypes in losses and targets**

| File | Change |
|---|---|
| `mmcv/models/dense_heads/orion_head.py`, `orion_head_map.py` | `bbox_targets` / `bbox_weights` are allocated in fp32, so fp32 ground truth can be written into them while the predictions are bf16 |
| `mmcv/models/detectors/base.py` | `_parse_losses`: `nan_to_num` on each loss term, and losses are cast to fp32 before `dist.all_reduce`. Under autocast a loss can be bf16 on one rank and fp32 on another, and the dtype mismatch deadlocks DDP |

**Start-up and environment**

| File | Change |
|---|---|
| `mmcv/utils/llava_llama.py` | `resize_token_embeddings(..., mean_resizing=False)`. The default covariance-based init of the new tokens stalled start-up for about 30 minutes on a busy CPU, and its result is overwritten anyway |
| `mmcv/models/utils/attention.py` | imports the flash-attn **1.x** entry point `flash_attn_unpadded_kvpacked_func`, which the A100 / RTX 3090 machines have. Swap the two import lines back for flash-attn 2.x |
| `mmcv/utils/runner_utils.py` | NCCL timeout read from `NCCL_TIMEOUT_S` (default 3600 s) |
| `mmcv/ops/nms.py`, `mmcv/core/evaluation/mean_ap.py`, `mmcv/core/mask/structures.py`, `mmcv/core/visualizer/open3d_vis.py`, `mmcv/datasets/pipelines/transforms_3d.py`, `mmcv/models/utils/vis_utils.py` | `np.bool` / `np.int` / `np.long` aliases replaced (removed in NumPy 1.24) |
| `mmcv/datasets/map_utils/struct.py` | returns an empty tensor instead of asserting on frames with no map elements, which some selected scenes have |

If trajectories look quantised at inference, bf16's precision is a likely cause: it has 7
mantissa bits, fp16 has 10.

## 6. Checking that the style changes the plan

Sweep a constant style over the same frames and compare the trajectories. The config's test
set is `b2d_infos_val.pkl`, which this repo does not build, so point it at the training pkl:

```bash
cd StyleOrion
for s in -1.0 0.0 1.0; do
  FORCE_DRIVING_STYLE=$s PYTHONPATH=. python adzoo/orion/test.py \
      adzoo/orion/configs/orion_stage2_vae_style_ft.py \
      adzoo/orion/work_dirs/orion_stage2_vae_style_ft/iter_<N>.pth \
      --cfg-options data.test.ann_file=data/infos/b2d_infos_train_style_mixed.pkl \
      --out work_dirs/vae_sweep_$s.pkl
done
```

- `--out` writes the predicted ego trajectory, `ego_fut_cmd`, `fut_valid_flag` and
  `metric_results` per frame. Call `test.py` directly rather than through
  `orion_dist_eval.sh`, which adds `--eval bbox` and runs the detection-mAP path that fails
  on bf16 `.numpy()`.
- If the three runs give identical trajectories, the style is not reaching the planner. Check
  that the pkl has non-zero `driving_style` values and that the checkpoint was trained with
  `use_style_conditioning=True`.
- `STYLE_GUIDANCE=g` with `g > 1` strengthens the effect (classifier-free guidance). It works
  because the model was trained with `style_dropout=0.15`.
- The VAE samples its latent noise with `torch.randn`. `test.py` seeds the RNGs (`--seed`,
  default 0), so the runs of a sweep use the same seed.

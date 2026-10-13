# FLUX.1 Kontext → FLUX.2 Klein 4B Distillation — Implementation Plan & TODOs

Goal: distill the FLUX.1 Kontext 12B teacher (inference via the ComfyUI submodule — the
`KSampler` / `FluxGuidance` / `DualCLIPLoader` code paths) into the FLUX.2 Klein 4B student
(trained via musubi-tuner's `Flux2NetworkTrainer`). The teacher runs 15 denoising steps; the
student is trained to reproduce the teacher's trajectory in 4 steps.

Because the two models use **different VAEs and different text encoders**:

- Text-encoder outputs for *both* models are pre-cached (no TE at train time).
- Both VAEs are small enough to stay resident in VRAM at all times.
- The teacher DiT and student DiT are swapped in **chunks** (not per batch) to amortize
  transfer cost. Torch caches and dead tensors are cleaned at each phase boundary to avoid OOM.

## Current state (untracked WIP files)

| File | Status |
|---|---|
| `flux_kontext_distill_train.py` | root launcher, done |
| `src/musubi_tuner/flux_kontext_distill_train.py` | draft trainer — needs the fixes below |
| `src/musubi_tuner/distillation/comfyui_teacher_pipeline.py` | draft — needs rewrite (Step 1) |
| `src/musubi_tuner/distillation/trajectory_capture.py` | done |
| `src/musubi_tuner/distillation/loss.py` | done |

## Verified facts the implementation relies on

### musubi-tuner

- `_run_training_loop` is called **positionally** from `train()` (`trainer_base.py` ~L1375).
  Positional order: 0=args, 1=accelerator, 2=session_id, 3=training_started_at,
  4=train_dataset_group, **5=train_dataloader**, 6=current_epoch, **7=transformer**, 8=network,
  … 17=vae, 19=dit_dtype, 20=network_dtype (21 params total).
- `on_train_start(args, accelerator, network, transformer, optimizer)` is called **inside**
  `_run_training_loop` — i.e. *after* any dataloader wrapper built in an override of it.
- `process_batch(args, accelerator, transformer, network, batch, latents, noise,
  noise_scheduler, dit_dtype, network_dtype, vae, global_step) -> (loss, metrics)`.
  The loop does `latents = batch["latents"]; noise = torch.randn_like(latents)` first.
- `Flux2NetworkTrainer.call_dit` reads `batch["ctx_vec"]` and `batch["latents_control_0..n"]`,
  computes `target = noise - latents`, divides `timesteps` by 1000 internally, uses
  `guidance_vec = 1.0`. `scale_shift_latents` is identity for FLUX.2.
- `clean_memory_on_device` lives in `musubi_tuner.training.accelerator_setup`.
- Cache file naming: `{basename}_{W:04d}x{H:04d}_{arch}.safetensors` and
  `{basename}_{arch}_te.safetensors`. Arch short names: `fk` (Kontext), `f2k4b` (Klein 4B).
  Caches for different architectures **coexist** in the same directory.
- Batch keys after collation (`dataset/bucket.py` `BucketBatchManager.__getitem__`):
  - Kontext (`fk`): `latents` [B,16,H/8,W/8], `latents_control`, `t5_vec`, `clip_l_pooler`
  - FLUX.2 (`f2k4b`): `latents` [B,128,H/16,W/16], `latents_control_0..n`, `ctx_vec`
  - The batch dict carries **no item identity** — cross-arch joining must happen inside
    `__getitem__` via cache-path suffix rewriting.
- Kontext AE (`flux_models.AutoEncoder`): 16ch, /8 spatial; `encode` stores latents in
  **scaled space**: `z = 0.3611 * (z_raw - 0.1159)`; `decode` inverts. Pixels BCHW [-1,1].
- FLUX.2 AE (`flux2_models.AutoEncoder`): z_channels=32, 2×2 pixel-shuffle → 128ch, /16
  spatial, BatchNorm latent normalization. Pixels BCHW [-1,1].

### ComfyUI (submodules/ComfyUI)

- `comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
  latent_image, denoise=1.0, ..., sigmas=None, callback=None, disable_pbar=False, seed=None)`.
- The callback is invoked with **4 positional args**: `callback(step_i, denoised_x0, x,
  total_steps)` (`comfy/samplers.py` ~L997) — *not* a dict.
- Flux conditioning format:
  `positive = [[t5_vec, {"pooled_output": clip_l_pooler, "guidance": g, "reference_latents": [lat]}]]`
  — element 0 is the cross-attn tensor.
- comfy applies `process_latent_in` (×0.3611, −0.1159) to `reference_latents` and to a nonzero
  `latent_image` → these must be passed in **raw** (unscaled) latent space.
- The callback's `denoised` x0 is in **model (scaled) latent space** — identical to musubi's
  cached Kontext latent space, so it feeds `flux_models.AutoEncoder.decode` directly.
- `cfg=1.0` with `negative=[]` is safe (cfg1 optimization skips uncond entirely).
- Load the teacher with `comfy.sd.load_diffusion_model(path, model_options={"dtype": ...})`
  → returns a ready `ModelPatcher` (same path as the `UNETLoader` node). Do **not** wrap a
  musubi Flux instance in `model_base.Flux` (it expects a class + comfy's BCHW forward).
- Import only `comfy.sample`, `comfy.samplers`, `comfy.sd`, `comfy.model_management` — never
  `nodes` (pulls in the server stack). `comfy.cli_args` is import-safe (`args_parsing=False`).
- Sigma schedule: `comfy.samplers.calculate_sigmas(model_sampling, "simple", steps)` —
  `sigma_i ≠ 1 - i/n` because `ModelSamplingFlux` applies a shift. Always use real sigmas.

## Bugs in the current draft (must all be fixed)

### `comfyui_teacher_pipeline.py`

1. Callback uses dict protocol; must be `callback(step, x0, x, total_steps)`.
2. `_create_comfy_model_patcher` cannot work (class-vs-instance + incompatible forward
   signature; `comfy.supported_models` not imported). Replace with
   `comfy.sd.load_diffusion_model`.
3. Conditioning malformed: `clip_l_pooler` placed at element 0; invented `"cross_attn"` key.
4. Latent-space double scaling: cached (scaled) Kontext latents passed where comfy expects raw.
5. `latent_image=noisy_latent` — must be `zeros_like(noise)` for txt2img-from-noise.
6. `from nodes import KSampler` drags in the ComfyUI server stack; also unused.
7. Captured x0 latents accumulate on GPU for the whole chunk; must move to CPU in the callback.

### `flux_kontext_distill_train.py`

8. Teacher noise made with `randn_like(student_latents)` `[B,128,H/16,W/16]`; teacher needs
   `[B,16,H/8,W/8]`.
9. Reads `t5_vec` / `clip_l_pooler` / `latents_control` from the f2k4b batch — those keys only
   exist in the `fk` caches; the batch has no item identity to fetch them (see Step 2).
10. `ChunkedDistillationDataLoader` captures `teacher_pipeline=None` at construction because
    `on_train_start` runs later, inside `super()._run_training_loop`. Late-bind via a trainer
    reference. (Teacher loading itself is also still a TODO placeholder.)
11. `t = 1 - step_idx/num_steps` ignores the sigma shift; use real sigmas. Also
    `teacher_pipeline.num_steps` attribute doesn't exist.
12. `batch["student_targets"]` raises KeyError on fallback path; use `.get`.
13. Import `clean_memory_on_device` from `musubi_tuner.training.accelerator_setup`.
14. Naive `student.to("cpu")` conflicts with `--blocks_to_swap`; must reject/guard it.

## Design decisions

- **D1 — Teacher loading:** via ComfyUI's own `comfy.sd.load_diffusion_model` (the tested
  `UNETLoader`+`KSampler` path). Text conditioning comes from musubi's cached `t5_vec` /
  `clip_l_pooler` (equivalent to `DualCLIPLoader` + `CLIPTextEncodeFlux` output), so no text
  encoder is loaded at train time.
- **D2 — VAEs:** use musubi's AE classes for both (BCHW / [-1,1] on both sides; teacher x0 is
  already in musubi's scaled space). Both stay resident on GPU permanently
  (~0.2 GB Kontext + ~1 GB FLUX.2 bf16). No per-chunk VAE movement.
- **D3 — Timestep grid:** compute the student's own 4-step sigma grid (with its shift), map each
  teacher capture sigma to the nearest student grid point, and train only at those 4 sigmas
  (random choice per batch). Choose teacher capture indices whose sigmas best match the grid.
- **D4 — Control images:** teacher `reference_latents` derived by **unscaling the cached `fk`
  control latent** (`raw = z/0.3611 + 0.1159`) — mathematically identical to re-encoding pixels
  and free. Student control latents come from the `f2k4b` cache. (On-the-fly pixel encoding via
  the resident VAEs remains possible later for augmentation.)
- **D5 — Future option:** teacher trajectories are deterministic given a seed → a separate
  offline `flux_kontext_cache_distill_targets.py` script could precompute all targets once and
  remove all runtime swapping. Not in scope now; the pipeline class should be reusable for it.

## Implementation TODOs

### Step 0 — Environment & dataset prerequisites
- [ ] Install ComfyUI runtime deps into the training venv (`torchsde`, `psutil`, `einops`, `av`
      — from `submodules/ComfyUI/requirements.txt`; server extras not needed).
- [ ] One dataset TOML shared by all caching runs; all resolutions **multiples of 32** (LCM of
      Kontext ×16 and FLUX.2 ×32 bucketing) so both architectures produce identical bucket
      resolutions → identical cache basenames.
- [ ] Run 4 caching passes on the same directories:
      `flux_kontext_cache_latents`, `flux_kontext_cache_text_encoder_outputs` (arch `fk`);
      `flux_2_cache_latents --model_version klein-4b`,
      `flux_2_cache_text_encoder_outputs --model_version klein-4b` (arch `f2k4b`).

### Step 1 — Rewrite `ComfyUITeacherPipeline` (fixes 1–7)
- [ ] `__init__(unet_path, dtype, device)`: import only `comfy.{sample, samplers, sd,
      model_management}`; `comfy.sd.load_diffusion_model(unet_path, model_options={"dtype": dtype})`
      once; keep the `ModelPatcher` with `offload_device="cpu"`.
- [ ] Precompute `self.sigmas = comfy.samplers.calculate_sigmas(
      patcher.get_model_object("model_sampling"), "simple", num_steps)` and expose it.
- [ ] `generate_trajectory(t5_vec, clip_l_pooler, ref_latent_raw, noise_16ch, num_steps,
      capture_indices, guidance, seed)`:
  - `positive = [[t5_vec, {"pooled_output": clip_l_pooler, "guidance": guidance,
    "reference_latents": [ref_latent_raw]}]]`, `negative=[]`, `cfg=1.0`.
  - `latent_image = torch.zeros_like(noise)`.
  - `callback(step, x0, x, total_steps)`: if `step in capture_indices` →
    `captured[step] = x0.detach().to("cpu")`.
  - Return `{step: x0_cpu}` and `{step: sigma}` (x0 in scaled Kontext space).
- [ ] `load()` / `offload()` using `comfy.model_management.load_models_gpu([patcher])` and
      unpatch + `soft_empty_cache()` — device movement through comfy's lifecycle, not raw `.to()`.

### Step 2 — Cross-architecture batch join (fixes 9)
- [ ] New `src/musubi_tuner/distillation/dual_arch_batch.py`: wrap each dataset's
      `BucketBatchManager.__getitem__` after `prepare_for_training()`. Per `ItemInfo`, rewrite
      cache paths (`_f2k4b.safetensors → _fk.safetensors`, `_f2k4b_te.safetensors →
      _fk_te.safetensors`), `load_file`, merge with `teacher_` prefix
      (`teacher_latents_control`, `teacher_t5_vec`, `teacher_clip_l_pooler`) using the same
      suffix-strip/stack rules as `bucket.py`.
- [ ] Startup validation: scan all items, error listing any missing `fk` cache files.

### Step 3 — Rework the trainer (fixes 8, 10–14)
- [ ] Parser: `--distill_teacher_dit`, `--distill_teacher_dtype {bf16,fp8_e4m3fn}`,
      `--distill_teacher_vae`, `--distill_guidance` (default **2.5**, Kontext's native value),
      `--distill_chunk_batches`, `--distill_teacher_steps` (15), `--distill_student_steps` (4),
      `--distill_loss_type`, `--distill_seed_mode {fixed,per_item}`.
- [ ] `handle_model_specific_args`: raise if `blocks_to_swap > 0` (fix 14).
- [ ] `on_train_start`: build `ComfyUITeacherPipeline` (weights on CPU); load Kontext AE via
      `flux_utils.load_ae` and FLUX.2 AE via `flux2_utils.load_ae`; move both AEs to GPU once,
      never move them again; precompute student 4-step sigma grid + teacher-capture mapping (D3).
- [ ] `ChunkedDistillationDataLoader` holds a reference to the **trainer** and resolves
      `trainer.teacher_pipeline` etc. at iteration time (fix 10).
- [ ] Keep the positional-index `_run_training_loop` override (indices 5/7 verified) but assert
      `len(args_list) == 21` to fail loudly on upstream signature changes.

### Step 4 — Correct chunk processing
- [ ] Phase order per chunk:
  1. Student → CPU; `clean_memory_on_device`.
  2. `teacher_pipeline.load()`; per batch:
     - teacher noise `randn(B, 16, h*16//8, w*16//8)` from a seeded generator (fix 8),
     - `ref_raw = batch["teacher_latents_control"] / 0.3611 + 0.1159` (fix 4),
     - `generate_trajectory(...)` with `teacher_t5_vec` / `teacher_clip_l_pooler`; captures go
       straight to CPU (fix 7),
     - `del` per-batch GPU tensors; `soft_empty_cache()` every K batches.
  3. `teacher_pipeline.offload()`; `clean_memory_on_device`.
  4. VAE conversion (VAEs already resident): per capture, `pixels =
     kontext_ae.decode(x0_scaled)` → `clamp(-1,1)` → `flux2_ae.encode(pixels)` → store as
     `batch["student_targets"] = {sigma: cpu_tensor}` (**keyed by sigma**, fix 11). Micro-batch
     with `torch.no_grad()`, free pixel tensors immediately.
  5. Student → GPU; `clean_memory_on_device`.

### Step 5 — Correct `process_batch`
- [ ] `student_targets = batch.get("student_targets")`; if absent → log once, fall back to
      vanilla flow matching (fix 12).
- [ ] Pick a training sigma from the student 4-step grid (D3);
      `x_t = (1-σ)·x0_teacher + σ·noise`; `timesteps = σ·1000`; call inherited
      `Flux2NetworkTrainer.call_dit` with `latents=student_target` (target becomes
      `noise − x0_teacher`); drop the pointless `batch["latents"]` swap.
- [ ] Loss via `distillation_loss`; log `loss/distill` and `distill/sigma`.

### Step 6 — Validation
- [ ] Unit tests for `get_capture_indices` and sigma mapping (pure functions).
- [ ] Smoke run: 1 item, chunk=2, teacher_steps=4, captures=2; assert shapes at every seam
      (`[B,16,H/8,W/8]` → `[B,3,H,W]` → `[B,128,H/16,W/16]`) and visually verify a decoded
      `student_targets` entry matches the decoded teacher x0 (round-trip check of scaling fixes).
- [ ] VRAM audit: log `torch.cuda.max_memory_allocated` per phase.
      Budget: teacher bf16 ~24 GB (fp8 ~12 GB) + ~1.2 GB AEs resident;
      student phase: Klein 4B bf16 ~8 GB + LoRA optimizer/activations + 1.2 GB AEs.
- [ ] 500-step LoRA run: `loss/distill` decreasing; 4-step samples via
      `flux_2_generate_image.py` improve vs. baseline.

## Known residual risks

- Single-GPU only initially (multi-GPU would run redundant teachers; needs chunk sharding).
- Teacher work repeats every epoch — mitigated by large chunks; eliminated later by the D5
  offline target-caching script.
- Resume mid-chunk restarts the chunk (acceptable; document in CLI help).

---

## Dry-Run Feature — Implementation Plan

### Goal

A `--dry_run` flag that exercises the **entire** distillation pipeline end-to-end
(argument parsing → dataset construction → model init → dataloader wrapping →
chunk processing → teacher trajectory → VAE cross-conversion → student forward →
loss computation → optimizer step) with **zero GPU VRAM and near-zero RAM**, and
**no real weight files required on disk**.  The purpose is to verify that all the
connections, shape plumbing, dtype flows, and control paths are wired correctly
before committing hours of GPU time.

### Constraints

| Concern | Dry-run behaviour |
|---|---|
| GPU VRAM | 0 — everything on CPU with empty/tiny tensors |
| System RAM | <200 MB — all models use `init_empty_weights` or meta tensors |
| Weight files | Not required.  The flag synthesises metadata-only stubs |
| Cache files | Either skipped entirely or synthesised as tiny stubs |
| ComfyUI teacher | Fully mocked (no `comfy.*` import needed) |
| Training steps | Runs exactly 2 steps then exits |
| Checkpointing | Skipped |

### Architecture

One new module: `src/musubi_tuner/distillation/dry_run.py`.

The core idea: rather than threading `if dry_run` checks throughout the codebase,
we **monkey-patch** the specific loading functions at the boundary before calling
`FluxKontextDistillTrainer().train(args)`, then restore them afterwards.  This
keeps the real code paths untouched — the dry run literally runs the real code,
just with hollow models.

```
┌──────────────────────────────────────────────────────┐
│  CLI: --dry_run                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ dry_run.activate(args)                         │  │
│  │   1. Patch model loaders  → return empty shells│  │
│  │   2. Patch dataset/cache  → return stubs       │  │
│  │   3. Patch teacher pipeline → mock class       │  │
│  │   4. Patch training loop  → exit after 2 steps │  │
│  │   5. Force args overrides (max_train_steps=2)  │  │
│  └────────────────────────────────────────────────┘  │
│  FluxKontextDistillTrainer().train(args)             │
│  dry_run.deactivate()                                │
└──────────────────────────────────────────────────────┘
```

### Implementation steps

#### Step D0 — CLI flag & entry point

- Add `--dry_run` to `flux_kontext_distill_setup_parser()`.
- In `main()`, if `args.dry_run`:
  ```python
  from musubi_tuner.distillation.dry_run import activate_dry_run, deactivate_dry_run
  activate_dry_run(args)
  try:
      FluxKontextDistillTrainer().train(args)
  finally:
      deactivate_dry_run()
  ```
- Force `args.max_train_steps = 2`, `args.save_every_n_steps = 999999`,
  `args.sample_every_n_steps = 999999`, `args.distill_chunk_batches = 2`,
  `args.max_data_loader_n_workers = 0`.

#### Step D1 — Mock model loaders (zero-weight shells)

**What to patch** (all in `dry_run.py` via `unittest.mock.patch`):

| Original function | Dry-run replacement |
|---|---|
| `flux2_utils.load_flow_model` | Build `Flux2(...)` with `init_empty_weights()`, fill with tiny random params on CPU |
| `flux2_utils.load_ae` | Build `AutoEncoder(...)` with `init_empty_weights()`, same |
| `flux_utils.load_ae` | Build `AutoEncoder(...)` with `init_empty_weights()`, same |
| `comfy.sd.load_diffusion_model` | Not called — teacher pipeline is mocked (D3) |

The "fill with tiny random params" step is needed because `init_empty_weights`
produces `meta`-device tensors which cannot participate in any operation.  The
replacement:
```python
def _materialize_empty(model):
    """Replace meta tensors with tiny CPU float32 tensors."""
    for name, param in list(model.named_parameters()):
        if param.device == torch.device("meta"):
            # Use shape [1] for every dim > 1 to save RAM
            tiny_shape = tuple(1 for _ in param.shape)
            new_param = torch.nn.Parameter(
                torch.zeros(tiny_shape, dtype=param.dtype, device="cpu"),
                requires_grad=param.requires_grad,
            )
            # assign via set_attr chain
            _set_param(model, name, new_param)
```

This gives us models that are structurally correct (same module tree, same
parameter names) but use ~0 RAM.

**Problem:** Operations on `[1,1,1,1]` tensors produce different shapes than
`[B,C,H,W]` — the forward pass would crash on shape mismatches.

**Solution:** We don't need the forward pass to produce correct *values*, just
not crash.  Two options:

- **(A) Full-shape zeros on CPU** — use the real shapes but `torch.zeros`.
  RAM cost: ~8 GB for Klein 4B bf16 + ~24 GB for Kontext.  Too much.
- **(B) Patch `torch.nn.Module.__call__` to return correctly-shaped zeros** —
  intercept the model forward, inspect input shapes, return a zeros tensor
  of the expected output shape.  Requires knowing each model's shape contract.
  Fragile.
- **(C) Patch at the trainer boundary, not the model** — replace `call_dit`,
  `teacher_vae.decode`, `student_vae.encode`, and teacher `generate_trajectory`
  with shape-aware stubs that return zeros of the correct shape.  These are
  **4 narrow seams** where input→output shape mapping is well-defined.

**Decision: (C)** — patch 4 call-sites, not the entire model stack.

#### Step D2 — Mock dataset & caches

The real dataset loader (`_build_dataset`) reads a TOML config and scans for
cache files.  In dry-run mode we need batches without real cache files.

**Approach:** Create a minimal in-memory dataset that yields synthetic batches
with the correct keys and shapes:

```python
class DryRunBatchIterator:
    """Yields synthetic batches matching the dual-arch distillation format."""
    def __init__(self, batch_size=1, height=512, width=512, num_batches=4):
        ...
    def __len__(self):
        return self.num_batches
    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                # Student (f2k4b) keys
                "latents": torch.zeros(B, 128, H//16, W//16),
                "latents_control_0": torch.zeros(B, 128, H//16, W//16),
                "ctx_vec": torch.zeros(B, 512, 15360),
                # Teacher (fk) keys (from dual_arch_batch wrapper)
                "teacher_latents": torch.zeros(B, 16, H//8, W//8),
                "teacher_latents_control": torch.zeros(B, 16, H//8, W//8),
                "teacher_t5_vec": torch.zeros(B, 512, 4096),
                "teacher_clip_l_pooler": torch.zeros(B, 768),
            }
```

**Patch target:** Replace `_build_dataset` to return a `DryRunDatasetGroup`
wrapper, and `_build_optimizer_and_dataloader` to use the synthetic dataloader.
Or more simply: patch `train_dataset_group` inside `_run_training_loop` to skip
cache validation, and inject the synthetic dataloader there (the distillation
trainer already overrides `_run_training_loop` and replaces the dataloader).

#### Step D3 — Mock teacher pipeline

Replace `ComfyUITeacherPipeline` with a mock that:
- `__init__`: no-op (no ComfyUI import, no weight loading).
- `sigmas`: returns `torch.linspace(1, 0, num_steps + 1)`.
- `load()` / `offload()`: no-op.
- `generate_trajectory(...)`: returns synthetic `captured_x0` and
  `captured_sigmas` with correct shapes (derived from input noise shape).

```python
class DryRunTeacherPipeline:
    def __init__(self, num_steps=15, **kwargs):
        self.num_steps = num_steps
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)

    def load(self): pass
    def offload(self): pass

    def generate_trajectory(self, t5_vec, clip_l_pooler, ref_latent_raw,
                            noise, capture_indices, guidance=2.5, seed=42):
        captured_x0 = {}
        captured_sigmas = {}
        for idx in capture_indices:
            captured_x0[idx] = torch.zeros_like(noise).cpu()
            captured_sigmas[idx] = self.sigmas[idx].item()
        return captured_x0, captured_sigmas
```

**Patch:** In `FluxKontextDistillTrainer.on_train_start`, the teacher pipeline
construction is `self.teacher_pipeline = ComfyUITeacherPipeline(...)`.  Patch
`ComfyUITeacherPipeline` at the module level before `train()` is called.

#### Step D4 — Mock VAE encode/decode

The VAE cross-conversion in `_process_chunk` calls:
- `teacher_vae.decode(x0_gpu)` → should return `[B, 3, H*8, W*8]`
- `student_vae.encode(pixels)` → should return `[B, 128, H//16, W//16]`

**Patch:** After `on_train_start` loads the (empty-weight) VAEs, replace their
`encode`/`decode` methods:

```python
def _mock_kontext_decode(self, z):
    B, C, H, W = z.shape  # [B, 16, H/8, W/8]
    return torch.zeros(B, 3, H * 8, W * 8, device=z.device, dtype=z.dtype)

def _mock_flux2_encode(self, pixels):
    B, C, H, W = pixels.shape  # [B, 3, H, W]
    return torch.zeros(B, 128, H // 16, W // 16, device=pixels.device, dtype=pixels.dtype)
```

Alternatively, since we're already mocking the teacher pipeline (D3) to return
zeros, and the VAE decode/encode just transforms zeros → zeros, the real VAEs
would work too *if* they had real weights.  But since we're using empty shells,
we must mock.

#### Step D5 — Mock student DiT forward (`call_dit`)

The `process_batch` calls `self.call_dit(...)` which runs the student DiT
forward.  With empty-weight models, this would crash.

**Patch:** Override `call_dit` in the trainer to return a synthetic `DiTOutput`:

```python
def _dry_run_call_dit(self, args, accelerator, transformer, latents, batch,
                      noise, noisy_model_input, timesteps, network_dtype, **kw):
    # pred and target must match shapes for loss computation
    target = noise - latents
    pred = torch.zeros_like(target)
    return DiTOutput(pred=pred, target=target)
```

#### Step D6 — Arg force-overrides and guard rails

In `activate_dry_run(args)`:
```python
args.max_train_steps = 2
args.distill_chunk_batches = 2
args.save_every_n_steps = 999999
args.sample_every_n_steps = 999999
args.max_data_loader_n_workers = 0
args.persistent_data_loader_workers = False
args.save_state = False
args.save_state_on_train_end = False
# Disable compile (no point in dry run, and it's slow)
args.compile = False
```

#### Step D7 — Patching strategy (putting it all together)

`dry_run.py` exposes two functions:

```python
_patches = []  # list of active unittest.mock._patch objects

def activate_dry_run(args):
    """Install all monkey-patches for a zero-resource dry run."""
    _force_args(args)

    # D1: model loaders
    _patches.append(patch("musubi_tuner.flux_2.flux2_utils.load_flow_model", _mock_load_flow_model))
    _patches.append(patch("musubi_tuner.flux_2.flux2_utils.load_ae", _mock_load_flux2_ae))
    _patches.append(patch("musubi_tuner.flux.flux_utils.load_ae", _mock_load_kontext_ae))

    # D2: dataset — patch _build_dataset to return synthetic data
    _patches.append(patch.object(
        FluxKontextDistillTrainer, "_build_dataset", _dry_run_build_dataset
    ))

    # D3: teacher pipeline
    _patches.append(patch(
        "musubi_tuner.flux_kontext_distill_train.ComfyUITeacherPipeline",
        DryRunTeacherPipeline,
    ))

    # D5: student forward
    _patches.append(patch.object(
        FluxKontextDistillTrainer, "call_dit", _dry_run_call_dit
    ))

    for p in _patches:
        p.start()

def deactivate_dry_run():
    for p in reversed(_patches):
        p.stop()
    _patches.clear()
```

#### Step D8 — Validation & output

At the end of a dry run (after the 2 steps complete), print a summary:
```
=== DRY RUN COMPLETE ===
  Steps executed:    2
  Batch keys seen:   latents, ctx_vec, teacher_t5_vec, ...
  Student input:     [1, 128, 32, 32]  (bf16)
  Teacher noise:     [1, 16, 64, 64]
  Teacher x0 caps:   4 captures at sigmas [0.93, 0.64, 0.36, 0.07]
  VAE decode→encode: [1,16,64,64] → [1,3,512,512] → [1,128,32,32]
  Loss computed:     0.0000 (mse)
  Peak RAM:          ~150 MB
  GPU VRAM:          0 MB
All connections verified OK.
```

This is implemented via a hook in `process_batch` that records shapes, and a
summary printed in `deactivate_dry_run()`.

#### Step D9 — Tests

- Unit test: `test_dry_run_activates_and_deactivates` — patches are applied
  and restored cleanly.
- Unit test: `test_dry_run_mock_shapes` — verify that `DryRunTeacherPipeline`,
  mock VAE encode/decode, and mock `call_dit` all produce correct output shapes.
- Integration test (no GPU): Run the full dry-run end-to-end with a synthetic
  dataset TOML and verify it completes without error.

### Execution order

| Step | Effort | Dependencies |
|---|---|---|
| D0 — CLI flag | Small | None |
| D6 — Arg overrides | Small | D0 |
| D3 — Mock teacher | Small | None |
| D4 — Mock VAE | Small | None |
| D5 — Mock call_dit | Small | None |
| D1 — Mock model loaders | Medium | None |
| D2 — Mock dataset/caches | Medium | None |
| D7 — Wiring (dry_run.py) | Medium | D0–D6 |
| D8 — Summary output | Small | D7 |
| D9 — Tests | Medium | D7 |

Total: ~1 new file (`dry_run.py`), ~3 lines changed in `flux_kontext_distill_train.py`,
~1 new test file.  Zero changes to `trainer_base.py` or any upstream code.

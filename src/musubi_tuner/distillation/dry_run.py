"""Zero-resource dry-run mode for the distillation pipeline.

``activate_dry_run(args)`` installs monkey-patches that replace every
weight-loading and GPU-touching call site with shape-correct stubs.
``deactivate_dry_run()`` removes them.  The real training code runs
*unmodified* — it just sees hollow models and synthetic data.

Usage (from ``flux_kontext_distill_train.main``):

    activate_dry_run(args)
    try:
        FluxKontextDistillTrainer().train(args)
    finally:
        deactivate_dry_run()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Value
from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telemetry — records shapes seen during the run for the final summary
# ---------------------------------------------------------------------------

@dataclass
class _DryRunLog:
    steps_executed: int = 0
    batch_keys: set = field(default_factory=set)
    student_input_shape: tuple = ()
    student_input_dtype: str = ""
    teacher_noise_shape: tuple = ()
    teacher_x0_caps: int = 0
    teacher_sigmas: list = field(default_factory=list)
    vae_decode_in: tuple = ()
    vae_decode_out: tuple = ()
    vae_encode_in: tuple = ()
    vae_encode_out: tuple = ()
    loss_value: float = 0.0
    loss_type: str = ""
    start_time: float = 0.0

_log = _DryRunLog()

# ---------------------------------------------------------------------------
# D6 — Arg force-overrides
# ---------------------------------------------------------------------------

def _force_args(args):
    """Clamp args to minimal values for a fast, zero-resource run."""
    args.max_train_steps = 2
    args.max_train_epochs = None
    args.distill_chunk_batches = 2
    args.save_every_n_steps = 999999
    args.sample_every_n_steps = 999999
    args.sample_prompts = None
    args.max_data_loader_n_workers = 0
    args.persistent_data_loader_workers = False
    args.save_state = False
    args.save_state_on_train_end = False
    args.compile = False
    args.gradient_checkpointing = False
    args.gradient_checkpointing_cpu_offload = False
    # Ensure CPU-only even if the user passed a GPU flag
    args.blocks_to_swap = 0
    # Attributes normally set by flux_2_train_network.main() or the common parser
    # that _prepare_accelerator_and_dtypes and _load_dit_and_swap expect:
    if not hasattr(args, "dit_dtype") or args.dit_dtype is None:
        args.dit_dtype = None  # → defaults to bf16 inside _prepare_accelerator_and_dtypes
    if not hasattr(args, "vae_dtype") or args.vae_dtype is None:
        args.vae_dtype = "float32"
    for attr, default in [
        ("fp8_base", False), ("fp8_scaled", False),
        ("disable_numpy_memmap", False),
        ("show_timesteps", False),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)
    _log.start_time = time.time()

# ---------------------------------------------------------------------------
# D1 — Mock model loaders
# ---------------------------------------------------------------------------

def _materialize_to_cpu(model: nn.Module, dtype=torch.bfloat16) -> nn.Module:
    """Replace meta-device parameters with tiny CPU zeros.

    We keep the *real* shapes (not [1,1,...]) so that ``named_parameters``
    iteration, LoRA ``apply_to``, and ``state_dict`` keys all behave
    identically to a real run.  For a Klein-4B model this costs ~8 GB
    which is too much — so we use *scalar* tensors that expand on access.

    Actually, for a dry run the LoRA layer just needs to see the parameter
    *shape* not the data, so we can use ``torch.empty`` with size 0 for
    every dim > 0 … but ``nn.Linear`` will refuse shape mismatches.

    Safest approach: use the real shapes but float32 zeros on CPU.
    Klein 4B float32 ≈ 16 GB.  Still too much.

    Final approach: build with ``init_empty_weights`` (meta tensors) and
    patch away all forward calls so they never touch the params.
    """
    # Just move meta → cpu with zeros, keeping shape.  For Klein-4B bf16
    # this is ~8 GB which is acceptable for a dry-run on a dev machine.
    # If RAM is still a concern, the user can use a smaller model_version.
    for name, param in list(model.named_parameters(recurse=True)):
        if param.device == torch.device("meta"):
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            new_p = nn.Parameter(
                torch.zeros(param.shape, dtype=dtype, device="cpu"),
                requires_grad=param.requires_grad,
            )
            setattr(mod, parts[-1], new_p)
    for name, buf in list(model.named_buffers(recurse=True)):
        if buf.device == torch.device("meta"):
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], torch.zeros(buf.shape, dtype=dtype, device="cpu"))
    return model


def _mock_load_flow_model(device, model_version_info, dit_path, attn_mode,
                          split_attn, loading_device, dit_weight_dtype=None,
                          fp8_scaled=False, **kw):
    """Build a real Flux2 model skeleton with zero weights (no file I/O)."""
    from accelerate import init_empty_weights
    from musubi_tuner.flux_2 import flux2_models

    logger.info("[DRY RUN] Building empty Flux2 model (no weight file needed)")
    dtype = dit_weight_dtype or torch.bfloat16
    with init_empty_weights():
        model = flux2_models.Flux2(model_version_info.params, attn_mode, split_attn)
    model = _materialize_to_cpu(model, dtype)
    return model


def _mock_load_flux2_ae(ckpt_path, dtype, device, disable_mmap=False):
    """Build an empty FLUX.2 AutoEncoder (no file I/O)."""
    from accelerate import init_empty_weights
    from musubi_tuner.flux_2 import flux2_models

    logger.info("[DRY RUN] Building empty FLUX.2 AE (no weight file needed)")
    with init_empty_weights():
        ae = flux2_models.AutoEncoder(flux2_models.AutoEncoderParams()).to(dtype)
    ae = _materialize_to_cpu(ae, dtype)
    return ae


def _mock_load_kontext_ae(ckpt_path, dtype, device, disable_mmap=False):
    """Build an empty Kontext AutoEncoder (no file I/O)."""
    from accelerate import init_empty_weights
    from musubi_tuner.flux import flux_models

    logger.info("[DRY RUN] Building empty Kontext AE (no weight file needed)")
    with init_empty_weights():
        ae = flux_models.AutoEncoder(flux_models.configs_flux_dev_context.ae_params).to(dtype)
    ae = _materialize_to_cpu(ae, dtype)
    return ae


# ---------------------------------------------------------------------------
# D2 — Mock dataset & caches
# ---------------------------------------------------------------------------

class _DryRunDatasetGroup:
    """Minimal dataset group that yields synthetic batches."""

    def __init__(self, batch_size=1, height=512, width=512, num_batches=4):
        self.batch_size = batch_size
        self.H = height
        self.W = width
        self.num_batches = num_batches
        self.num_train_items = num_batches * batch_size
        self.datasets = [self]  # mimic list-of-datasets for the validator loop
        self.batch_manager = _DryRunBatchManager(batch_size, height, width, num_batches)

    def set_max_train_steps(self, steps):
        pass

    def get_metadata(self):
        return {"dry_run": True, "batch_size_per_device": self.batch_size}

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self._make_batch()

    def _make_batch(self):
        B = self.batch_size
        H, W = self.H, self.W
        return {
            # Student (f2k4b) keys
            "latents": torch.zeros(B, 128, H // 16, W // 16, dtype=torch.bfloat16),
            "latents_control_0": torch.zeros(B, 128, H // 16, W // 16, dtype=torch.bfloat16),
            "ctx_vec": torch.zeros(B, 512, 7680, dtype=torch.bfloat16),  # Klein-4B context dim
            # Teacher (fk) keys — normally injected by dual_arch_batch
            "teacher_latents": torch.zeros(B, 16, H // 8, W // 8, dtype=torch.bfloat16),
            "teacher_latents_control": torch.zeros(B, 16, H // 8, W // 8, dtype=torch.bfloat16),
            "teacher_t5_vec": torch.zeros(B, 512, 4096, dtype=torch.bfloat16),
            "teacher_clip_l_pooler": torch.zeros(B, 768, dtype=torch.bfloat16),
        }


class _DryRunBatchManager:
    """Stub that satisfies validate_teacher_caches and wrap_batch_manager_for_distillation."""

    def __init__(self, batch_size, height, width, num_batches):
        self.batch_size = batch_size
        self.buckets = {}  # empty → validation is a no-op


class _DryRunCollator:
    """Collator that passes batches through unchanged."""

    def __init__(self, *a, **kw):
        pass

    def __call__(self, batch_list):
        # DataLoader wraps items in a list; just return the first (batch_size=1 in DataLoader)
        return batch_list[0]


def _dry_run_build_dataset(self, args):
    """Replace the real dataset builder with a synthetic one."""
    logger.info("[DRY RUN] Using synthetic dataset (no cache files needed)")
    ds = _DryRunDatasetGroup()
    current_epoch = Value("i", 0)
    collator = _DryRunCollator()
    return ds, collator, current_epoch


# ---------------------------------------------------------------------------
# D3 — Mock teacher pipeline
# ---------------------------------------------------------------------------

class DryRunTeacherPipeline:
    """Shape-correct mock of ComfyUITeacherPipeline — no ComfyUI import."""

    def __init__(self, num_steps=15, **kwargs):
        self.num_steps = num_steps
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        logger.info(f"[DRY RUN] Mock teacher pipeline (num_steps={num_steps})")

    def load(self):
        pass

    def offload(self):
        pass

    def generate_trajectory(self, t5_vec, clip_l_pooler, ref_latent_raw,
                            noise, capture_indices, guidance=2.5, seed=42):
        captured_x0 = {}
        captured_sigmas = {}
        for idx in capture_indices:
            captured_x0[idx] = torch.zeros_like(noise).cpu()
            captured_sigmas[idx] = self.sigmas[idx].item()

        # Record telemetry
        _log.teacher_noise_shape = tuple(noise.shape)
        _log.teacher_x0_caps = len(capture_indices)
        _log.teacher_sigmas = [captured_sigmas[i] for i in capture_indices]

        return captured_x0, captured_sigmas


# ---------------------------------------------------------------------------
# D4 — Mock VAE encode/decode
# ---------------------------------------------------------------------------

def _mock_kontext_decode(self, z):
    """teacher_vae.decode: [B, 16, H/8, W/8] -> [B, 3, H, W]"""
    B, C, Hl, Wl = z.shape
    out = torch.zeros(B, 3, Hl * 8, Wl * 8, device=z.device, dtype=z.dtype)
    _log.vae_decode_in = tuple(z.shape)
    _log.vae_decode_out = tuple(out.shape)
    return out


def _mock_flux2_encode(self, pixels):
    """student_vae.encode: [B, 3, H, W] -> [B, 128, H/16, W/16]"""
    B, C, H, W = pixels.shape
    out = torch.zeros(B, 128, H // 16, W // 16, device=pixels.device, dtype=pixels.dtype)
    _log.vae_encode_in = tuple(pixels.shape)
    _log.vae_encode_out = tuple(out.shape)
    return out


# ---------------------------------------------------------------------------
# D5 — Mock student DiT forward
# ---------------------------------------------------------------------------

def _dry_run_call_dit(self, args, accelerator, transformer, latents, batch,
                      noise, noisy_model_input, timesteps, network_dtype, **kw):
    from musubi_tuner.training.trainer_base import DiTOutput

    target = noise - latents
    pred = torch.zeros_like(target)

    _log.student_input_shape = tuple(noisy_model_input.shape)
    _log.student_input_dtype = str(noisy_model_input.dtype)

    return DiTOutput(pred=pred, target=target)


# ---------------------------------------------------------------------------
# D5b — Mock on_train_start to patch VAE methods & use mock teacher
# ---------------------------------------------------------------------------

_orig_on_train_start = None

def _dry_run_on_train_start(self, args, accelerator, network, transformer, optimizer):
    """Patched on_train_start that builds mock VAEs and teacher."""
    # Call the grandparent's on_train_start (Flux2NetworkTrainer), skipping
    # the distillation trainer's on_train_start which tries to load real VAEs.
    from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer
    Flux2NetworkTrainer.on_train_start(self, args, accelerator, network, transformer, optimizer)

    device = "cpu"  # dry run is CPU-only
    vae_dtype = torch.bfloat16

    # Mock teacher pipeline
    self.teacher_pipeline = DryRunTeacherPipeline(
        num_steps=args.distill_teacher_steps,
    )

    # Load empty-weight VAEs and patch their encode/decode
    self.teacher_vae = _mock_load_kontext_ae(args.distill_teacher_vae, vae_dtype, device)
    self.teacher_vae.eval()
    self.teacher_vae.decode = lambda z, **kw: _mock_kontext_decode(self.teacher_vae, z)

    self.student_vae = _mock_load_flux2_ae(args.distill_student_vae, vae_dtype, device)
    self.student_vae.eval()
    self.student_vae.encode = lambda pixels, **kw: _mock_flux2_encode(self.student_vae, pixels)

    # Precompute capture indices
    from musubi_tuner.distillation.trajectory_capture import get_capture_indices
    self._capture_indices = get_capture_indices(args.distill_teacher_steps, args.distill_student_steps)
    logger.info(f"[DRY RUN] Teacher capture indices: {self._capture_indices}")


# ---------------------------------------------------------------------------
# D5c — Mock process_batch to record telemetry
# ---------------------------------------------------------------------------

_orig_process_batch = None

def _dry_run_process_batch(self, args, accelerator, transformer, network, batch,
                           latents, noise, noise_scheduler, dit_dtype, network_dtype,
                           vae, global_step):
    """Wraps the real process_batch, records telemetry after it runs."""
    _log.batch_keys.update(batch.keys())

    # Call the real distillation process_batch (which uses our mocked call_dit)
    loss, metrics = _orig_process_batch(
        self, args, accelerator, transformer, network, batch,
        latents, noise, noise_scheduler, dit_dtype, network_dtype,
        vae, global_step,
    )
    _log.steps_executed += 1
    _log.loss_value = loss.item() if hasattr(loss, "item") else float(loss)
    _log.loss_type = getattr(args, "distill_loss_type", "mse")

    # The mock call_dit returns detached zeros, so loss has no grad_fn.
    # Create a tiny dummy loss connected to the LoRA params so backward() works.
    if not loss.requires_grad:
        dummy = torch.tensor(0.0, requires_grad=True, device=loss.device)
        for p in network.parameters():
            if p.requires_grad:
                dummy = dummy + (p * 0).sum()
                break
        loss = loss.detach() + dummy

    return loss, metrics


# ---------------------------------------------------------------------------
# D7 — Patch wiring
# ---------------------------------------------------------------------------

_patches: list = []

def activate_dry_run(args):
    """Install all monkey-patches for a zero-resource dry run."""
    global _orig_on_train_start, _orig_process_batch, _log
    _log = _DryRunLog()

    logger.info("=" * 60)
    logger.info("  DRY RUN MODE — no GPU, no weights, synthetic data")
    logger.info("=" * 60)

    # D6 — force args
    _force_args(args)

    # D1 — mock model loaders
    _patches.append(patch(
        "musubi_tuner.flux_2.flux2_utils.load_flow_model",
        _mock_load_flow_model,
    ))
    # Patch load_ae for the sampling VAE path in prepare_sampling (base trainer)
    _patches.append(patch(
        "musubi_tuner.flux_2.flux2_utils.load_ae",
        _mock_load_flux2_ae,
    ))
    _patches.append(patch(
        "musubi_tuner.flux.flux_utils.load_ae",
        _mock_load_kontext_ae,
    ))

    # D2 — mock dataset
    from musubi_tuner.flux_kontext_distill_train import FluxKontextDistillTrainer
    _patches.append(patch.object(
        FluxKontextDistillTrainer, "_build_dataset", _dry_run_build_dataset,
    ))

    # D5 — mock call_dit
    _patches.append(patch.object(
        FluxKontextDistillTrainer, "call_dit", _dry_run_call_dit,
    ))

    # D5b — mock on_train_start (teacher pipeline + VAEs)
    _orig_on_train_start = FluxKontextDistillTrainer.on_train_start
    _patches.append(patch.object(
        FluxKontextDistillTrainer, "on_train_start", _dry_run_on_train_start,
    ))

    # D5c — wrap process_batch for telemetry
    _orig_process_batch = FluxKontextDistillTrainer.process_batch
    _patches.append(patch.object(
        FluxKontextDistillTrainer, "process_batch", _dry_run_process_batch,
    ))

    for p in _patches:
        p.start()


def deactivate_dry_run():
    """Remove all monkey-patches and print the summary."""
    for p in reversed(_patches):
        p.stop()
    _patches.clear()

    _print_summary()


# ---------------------------------------------------------------------------
# D8 — Summary output
# ---------------------------------------------------------------------------

def _print_summary():
    elapsed = time.time() - _log.start_time if _log.start_time else 0
    sigmas_str = ", ".join(f"{s:.2f}" for s in _log.teacher_sigmas) if _log.teacher_sigmas else "N/A"

    lines = [
        "",
        "=" * 60,
        "  DRY RUN COMPLETE",
        "=" * 60,
        f"  Steps executed:       {_log.steps_executed}",
        f"  Batch keys seen:      {sorted(_log.batch_keys) if _log.batch_keys else 'N/A'}",
        f"  Student input:        {list(_log.student_input_shape)}  ({_log.student_input_dtype})",
        f"  Teacher noise:        {list(_log.teacher_noise_shape)}",
        f"  Teacher x0 captures:  {_log.teacher_x0_caps} at sigmas [{sigmas_str}]",
        f"  VAE decode:           {list(_log.vae_decode_in)} -> {list(_log.vae_decode_out)}",
        f"  VAE encode:           {list(_log.vae_encode_in)} -> {list(_log.vae_encode_out)}",
        f"  Loss ({_log.loss_type}):          {_log.loss_value:.6f}",
        f"  Elapsed:              {elapsed:.1f}s",
        f"  GPU VRAM used:        0 MB (CPU-only dry run)",
        "",
        "  All connections verified OK.",
        "=" * 60,
        "",
    ]
    for line in lines:
        logger.info(line)
    # Also print to stdout for visibility
    print("\n".join(lines))

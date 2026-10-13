import argparse
import logging
import random
import torch
from typing import Optional

from accelerate import Accelerator

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import (
    DiTOutput,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.distillation.comfyui_teacher_pipeline import ComfyUITeacherPipeline
from musubi_tuner.distillation.dual_arch_batch import (
    validate_teacher_caches,
    wrap_batch_manager_for_distillation,
)
from musubi_tuner.distillation.trajectory_capture import get_capture_indices
from musubi_tuner.distillation.loss import distillation_loss

logger = logging.getLogger(__name__)

# Kontext AE scaling constants (z_scaled = 0.3611 * (z_raw - 0.1159))
_KONTEXT_SCALE = 0.3611
_KONTEXT_SHIFT = 0.1159


def _unscale_kontext_latent(z_scaled: torch.Tensor) -> torch.Tensor:
    """Convert cached (scaled) Kontext latents to raw space for ComfyUI."""
    return z_scaled / _KONTEXT_SCALE + _KONTEXT_SHIFT


def _batch_to_cpu(batch: dict) -> dict:
    """Move all tensor values in a batch dict to CPU."""
    return {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _batch_to_device(batch: dict, device) -> dict:
    """Move all tensor values in a batch dict to *device*."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Chunked dataloader wrapper
# ---------------------------------------------------------------------------

class ChunkedDistillationDataLoader:
    """Wraps the training dataloader to process batches in chunks.

    For each chunk:
      1. Student DiT → CPU
      2. Teacher generates trajectories, captures x0 → CPU
      3. Teacher → CPU; VAE cross-conversion (both AEs stay on GPU)
      4. Student DiT → GPU

    The teacher pipeline and VAEs are resolved at iteration time via the
    trainer reference (late binding), because ``on_train_start`` runs inside
    ``_run_training_loop`` — after this wrapper is constructed (fix 10).
    """

    def __init__(self, orig_dataloader, trainer: "FluxKontextDistillTrainer"):
        self.orig_dataloader = orig_dataloader
        self.trainer = trainer

    def __len__(self):
        return len(self.orig_dataloader)

    def __iter__(self):
        args = self.trainer._distill_args
        chunk_size = args.distill_chunk_batches
        device = self.trainer._distill_accelerator.device

        iterator = iter(self.orig_dataloader)
        while True:
            chunk = []
            for _ in range(chunk_size):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                # Accelerate's DataLoaderShard places batches on GPU.
                # Holding chunk_size GPU batches (~2-3 GB) while the 12B
                # teacher is also resident would OOM.  Move to CPU now;
                # _process_chunk works on CPU tensors anyway (it sends
                # individual fields to GPU as needed), and we restore
                # device placement when yielding.
                chunk.append(_batch_to_cpu(batch))
            if not chunk:
                break

            self._process_chunk(chunk)

            for batch in chunk:
                yield _batch_to_device(batch, device)

    # ------------------------------------------------------------------
    # Core chunk processing (Steps 4 from the plan)
    # ------------------------------------------------------------------

    def _process_chunk(self, chunk: list[dict]):
        trainer = self.trainer
        args = trainer._distill_args
        accelerator = trainer._distill_accelerator
        device = accelerator.device

        teacher_pipeline: ComfyUITeacherPipeline = trainer.teacher_pipeline
        teacher_vae = trainer.teacher_vae  # Kontext AE — stays on GPU
        student_vae = trainer.student_vae  # FLUX.2 AE — stays on GPU
        capture_indices = trainer._capture_indices

        logger.info(f"Processing chunk of {len(chunk)} batches for distillation targets.")

        # --- Phase 1: offload student, run teacher ---
        student_unwrapped = accelerator.unwrap_model(trainer._student_transformer)
        student_unwrapped.to("cpu")
        clean_memory_on_device(device)

        teacher_pipeline.load()

        seed_base = args.distill_seed if args.distill_seed_mode == "fixed" else None
        guidance = args.distill_guidance

        for batch_i, batch in enumerate(chunk):
            # Teacher noise must be [B, 16, H/8, W/8] (fix 8).
            # The student latents are [B, 128, H/16, W/16]; derive teacher
            # spatial dims: H_teacher = H_student * 2, W_teacher = W_student * 2
            # because Kontext is /8 and FLUX.2 is /16.
            B = batch["latents"].shape[0]
            h_student = batch["latents"].shape[2]
            w_student = batch["latents"].shape[3]
            h_teacher = h_student * 2
            w_teacher = w_student * 2

            seed = seed_base if seed_base is not None else random.randint(0, 2**31)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise_16ch = torch.randn(B, 16, h_teacher, w_teacher, generator=gen, device="cpu")

            # Teacher conditioning from the dual-arch batch (fix 9)
            t5_vec = batch["teacher_t5_vec"].to(device)
            clip_l_pooler = batch["teacher_clip_l_pooler"].to(device)

            # Control latent: unscale from cached (scaled) to raw (fix 4)
            ref_latent_raw = None
            teacher_ctrl = batch.get("teacher_latents_control")
            if teacher_ctrl is not None:
                ref_latent_raw = _unscale_kontext_latent(teacher_ctrl).to(device)

            noise_16ch = noise_16ch.to(device)

            captured_x0, captured_sigmas = teacher_pipeline.generate_trajectory(
                t5_vec=t5_vec,
                clip_l_pooler=clip_l_pooler,
                ref_latent_raw=ref_latent_raw,
                noise=noise_16ch,
                capture_indices=capture_indices,
                guidance=guidance,
                seed=seed,
            )
            # captured_x0 values are already on CPU (fix 7)
            batch["_teacher_x0"] = captured_x0
            batch["_teacher_sigmas"] = captured_sigmas

            # Free per-batch GPU tensors
            del t5_vec, clip_l_pooler, ref_latent_raw, noise_16ch

        # --- Phase 2: offload teacher, VAE cross-conversion ---
        teacher_pipeline.offload()
        clean_memory_on_device(device)

        # VAEs are already on GPU (loaded once in on_train_start, never moved)
        logger.info("Phase 2: VAE cross-conversion (teacher x0 → student latent space)")
        for batch in chunk:
            captured_x0 = batch.pop("_teacher_x0")
            captured_sigmas = batch.pop("_teacher_sigmas")
            student_targets = {}  # keyed by sigma (fix 11)

            for step_idx, x0_cpu in captured_x0.items():
                sigma = captured_sigmas[step_idx]
                # ComfyUI's model_base.calculate_denoised casts to float32;
                # the teacher VAE expects its native dtype (bf16).
                vae_dtype = next(teacher_vae.parameters()).dtype
                x0_gpu = x0_cpu.to(device=device, dtype=vae_dtype)

                with torch.no_grad():
                    # Decode: scaled Kontext latent → pixels BCHW [-1,1]
                    pixels = teacher_vae.decode(x0_gpu)
                    pixels = pixels.clamp(-1.0, 1.0)
                    # Encode: pixels → FLUX.2 128ch latent
                    student_target = student_vae.encode(pixels)

                student_targets[sigma] = student_target.to("cpu")
                del x0_gpu, pixels, student_target

            batch["student_targets"] = student_targets

        # --- Phase 3: reload student ---
        clean_memory_on_device(device)
        student_unwrapped.to(device)
        clean_memory_on_device(device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FluxKontextDistillTrainer(Flux2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.teacher_pipeline: Optional[ComfyUITeacherPipeline] = None
        self.teacher_vae = None  # Kontext AE (16ch)
        self.student_vae = None  # FLUX.2 AE (128ch)

        # Set during _run_training_loop for late-binding by the dataloader
        self._distill_args = None
        self._distill_accelerator = None
        self._student_transformer = None

        # Precomputed in on_train_start
        self._capture_indices = None

    # ------------------------------------------------------------------
    # Argument handling
    # ------------------------------------------------------------------

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        super().handle_model_specific_args(args)

        # Reject blocks_to_swap — naively moving the student to CPU would
        # conflict with the block-swap bookkeeping (fix 14).
        blocks_to_swap = getattr(args, "blocks_to_swap", 0) or 0
        if blocks_to_swap > 0:
            raise ValueError(
                "Distillation trainer is incompatible with --blocks_to_swap "
                "(the student is moved to CPU between chunks). "
                "Set --blocks_to_swap 0 or omit it."
            )

    # ------------------------------------------------------------------
    # Model & pipeline initialisation
    # ------------------------------------------------------------------

    def on_train_start(
        self, args: argparse.Namespace, accelerator: Accelerator, network, transformer, optimizer
    ) -> None:
        super().on_train_start(args, accelerator, network, transformer, optimizer)

        device = accelerator.device
        vae_dtype = torch.bfloat16

        # --- Teacher pipeline (weights on CPU until chunk phase) ---
        teacher_dtype_str = getattr(args, "distill_teacher_dtype", "bf16")
        teacher_dtype = torch.float8_e4m3fn if teacher_dtype_str == "fp8_e4m3fn" else torch.bfloat16

        logger.info("Building teacher pipeline...")
        self.teacher_pipeline = ComfyUITeacherPipeline(
            unet_path=args.distill_teacher_dit,
            num_steps=args.distill_teacher_steps,
            dtype=teacher_dtype,
            device=str(device),
        )

        # --- VAEs: load once, keep on GPU permanently (D2) ---
        from musubi_tuner.flux import flux_utils
        from musubi_tuner.flux_2 import flux2_utils as f2_utils

        logger.info(f"Loading Kontext AE from {args.distill_teacher_vae}")
        self.teacher_vae = flux_utils.load_ae(args.distill_teacher_vae, vae_dtype, device)
        self.teacher_vae.eval()

        logger.info(f"Loading FLUX.2 AE from {args.distill_student_vae}")
        self.student_vae = f2_utils.load_ae(args.distill_student_vae, vae_dtype, device)
        self.student_vae.eval()

        # --- Precompute capture indices and student sigma grid (D3) ---
        num_teacher_steps = args.distill_teacher_steps
        num_student_steps = args.distill_student_steps

        self._capture_indices = get_capture_indices(num_teacher_steps, num_student_steps)
        logger.info(f"Teacher capture indices: {self._capture_indices}")

        # --- Validate teacher caches ---
        # This runs after prepare_for_training has built the batch_manager
        # on each dataset in the dataset group.
        # We'll validate and wrap in _run_training_loop where we have access
        # to the dataset group.

    # ------------------------------------------------------------------
    # Training loop override
    # ------------------------------------------------------------------

    def _run_training_loop(self, *args_list, **kwargs):
        # Verify positional arg count to fail loudly on upstream changes
        assert len(args_list) == 21, (
            f"Expected 21 positional args to _run_training_loop, got {len(args_list)}. "
            "The trainer_base.py signature may have changed."
        )

        args = args_list[0]
        accelerator = args_list[1]
        train_dataset_group = args_list[4]
        train_dataloader = args_list[5]
        transformer = args_list[7]

        # Store references for late-binding by ChunkedDistillationDataLoader (fix 10)
        self._distill_args = args
        self._distill_accelerator = accelerator
        self._student_transformer = transformer

        # Validate and wrap each dataset's batch_manager for dual-arch loading
        for dataset in train_dataset_group.datasets:
            bm = dataset.batch_manager
            validate_teacher_caches(bm)
            wrap_batch_manager_for_distillation(bm)

        # Wrap the dataloader
        chunked_dataloader = ChunkedDistillationDataLoader(
            orig_dataloader=train_dataloader,
            trainer=self,
        )

        new_args_list = list(args_list)
        new_args_list[5] = chunked_dataloader
        super()._run_training_loop(*new_args_list, **kwargs)

    # ------------------------------------------------------------------
    # Loss computation (Step 5)
    # ------------------------------------------------------------------

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        # Retrieve precomputed distillation targets (fix 12)
        student_targets = batch.get("student_targets")
        if student_targets is None or len(student_targets) == 0:
            logger.warning("No student_targets in batch — falling back to vanilla flow matching")
            return super().process_batch(
                args, accelerator, transformer, network, batch,
                latents, noise, noise_scheduler, dit_dtype, network_dtype, vae, global_step,
            )

        # Pick a random sigma from the available targets
        available_sigmas = list(student_targets.keys())
        sigma = random.choice(available_sigmas)
        student_target = student_targets[sigma]
        student_target = student_target.to(device=accelerator.device, dtype=network_dtype)

        # Construct noisy input at this sigma:
        #   x_t = (1 - sigma) * x0 + sigma * noise
        # where x0 = student_target (in FLUX.2 latent space)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = (1.0 - sigma) * student_target + sigma * noise

        # Timesteps for call_dit (which divides by 1000 internally):
        #   t_model = sigma * 1000 + 1  (matching the base trainer convention)
        # Use float32 for timesteps to avoid ~0.4% precision loss from bf16.
        bsize = latents.shape[0]
        timesteps = torch.full(
            (bsize,), sigma * 1000.0 + 1.0,
            device=accelerator.device, dtype=torch.float32,
        )

        # Temporarily swap batch["latents"] so call_dit computes the correct
        # target = noise - latents (where latents = student_target here)
        orig_latents = batch["latents"]
        batch["latents"] = student_target

        dit_out = self.call_dit(
            args, accelerator, transformer, student_target, batch,
            noise, noisy_model_input, timesteps, network_dtype,
        )

        batch["latents"] = orig_latents

        loss_type = getattr(args, "distill_loss_type", "mse")
        loss = distillation_loss(dit_out.pred, dit_out.target, loss_type=loss_type)
        loss = loss.mean()

        return loss, {"loss/distill": loss.item(), "distill/sigma": sigma}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def flux_kontext_distill_setup_parser() -> argparse.ArgumentParser:
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)

    # Distillation-specific arguments
    g = parser.add_argument_group("Distillation")
    g.add_argument("--distill_teacher_dit", type=str, required=True,
                   help="Path to FLUX.1 Kontext teacher DiT checkpoint (safetensors)")
    g.add_argument("--distill_teacher_dtype", type=str, default="bf16",
                   choices=["bf16", "fp8_e4m3fn"],
                   help="Precision for teacher DiT (default: bf16)")
    g.add_argument("--distill_teacher_vae", type=str, required=True,
                   help="Path to Kontext VAE (ae.safetensors)")
    g.add_argument("--distill_student_vae", type=str, required=True,
                   help="Path to FLUX.2 VAE (ae.safetensors)")
    g.add_argument("--distill_guidance", type=float, default=2.5,
                   help="Flux guidance value for teacher generation (default: 2.5)")
    g.add_argument("--distill_chunk_batches", type=int, default=100,
                   help="Number of batches per teacher-generation chunk (default: 100)")
    g.add_argument("--distill_teacher_steps", type=int, default=15,
                   help="Number of teacher denoising steps (default: 15)")
    g.add_argument("--distill_student_steps", type=int, default=4,
                   help="Number of student inference steps / sigma grid size (default: 4)")
    g.add_argument("--distill_loss_type", type=str, default="mse",
                   choices=["mse", "l1", "huber"],
                   help="Loss function for distillation (default: mse)")
    g.add_argument("--distill_seed_mode", type=str, default="per_item",
                   choices=["fixed", "per_item"],
                   help="Teacher noise seed: 'fixed' for deterministic, 'per_item' for random (default: per_item)")
    g.add_argument("--distill_seed", type=int, default=42,
                   help="Seed value when --distill_seed_mode=fixed (default: 42)")
    g.add_argument("--dry_run", action="store_true",
                   help="Run the full pipeline with mock models and synthetic data "
                        "(zero GPU VRAM, no weight files needed) to verify all connections")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = flux_kontext_distill_setup_parser()
        args = parser.parse_args()
        args = read_config_from_file(args, parser)

    if getattr(args, "dry_run", False):
        from musubi_tuner.distillation.dry_run import activate_dry_run, deactivate_dry_run
        activate_dry_run(args)
        try:
            FluxKontextDistillTrainer().train(args)
        finally:
            deactivate_dry_run()
    else:
        FluxKontextDistillTrainer().train(args)


if __name__ == "__main__":
    main()

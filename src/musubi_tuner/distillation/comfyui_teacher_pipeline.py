import sys
import os
import torch
import logging
from typing import Dict, List, Optional

from musubi_tuner.distillation.trajectory_capture import get_capture_indices

logger = logging.getLogger(__name__)

# comfy.* imports are deferred to _ensure_comfy() so this module stays
# importable without the ComfyUI submodule (tests, CLI parsing, etc.).
_comfy = None  # set to a namespace by _ensure_comfy()


class _ComfyNS:
    """Thin holder so call-sites read ``_comfy.sample`` etc."""
    pass


def _ensure_comfy():
    """Lazily add ComfyUI to sys.path and import the four modules we need."""
    global _comfy
    if _comfy is not None:
        return

    # Repo-local submodule: <repo>/submodules/ComfyUI
    # When running from the top-level launcher the file lives under src/,
    # so walk up to the repo root from __file__.
    for ancestor in [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "submodules", "ComfyUI"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "submodules", "ComfyUI"),
    ]:
        p = os.path.abspath(ancestor)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.model_management

    ns = _ComfyNS()
    ns.sample = comfy.sample
    ns.samplers = comfy.samplers
    ns.sd = comfy.sd
    ns.model_management = comfy.model_management
    _comfy = ns


class ComfyUITeacherPipeline:
    """Wraps ComfyUI's sampling path to run the FLUX.1 Kontext teacher and
    capture intermediate x0 predictions for distillation.

    The teacher model is loaded once via ``comfy.sd.load_diffusion_model``
    (the same path as the ``UNETLoader`` node) and kept on CPU between chunks.
    ``load()`` / ``offload()`` move the model through comfy's lifecycle so
    device management stays consistent with the rest of the framework.
    """

    def __init__(self, unet_path: str, num_steps: int = 15, dtype=torch.bfloat16, device="cuda"):
        _ensure_comfy()

        self.device = device
        self.num_steps = num_steps

        # Load teacher weights via ComfyUI (returns a ModelPatcher / CoreModelPatcher)
        model_options = {"dtype": dtype}
        logger.info(f"Loading teacher DiT from {unet_path} (dtype={dtype})")
        self.patcher = _comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        # Precompute the sigma schedule (respects ModelSamplingFlux shift)
        model_sampling = self.patcher.get_model_object("model_sampling")
        self.sigmas = _comfy.samplers.calculate_sigmas(model_sampling, "simple", num_steps)
        # sigmas is a 1-D tensor of length num_steps+1 (last element is 0)

    # ------------------------------------------------------------------
    # Device lifecycle
    # ------------------------------------------------------------------

    def load(self):
        """Move teacher to GPU through comfy's model management."""
        _comfy.model_management.load_models_gpu([self.patcher])

    def offload(self):
        """Move teacher back to CPU and free VRAM.

        Uses ``free_memory`` so that comfy's ``current_loaded_models`` list is
        updated properly (the LoadedModel entry is popped via ``model_unload``).
        Calling ``unpatch_model`` directly would leave a stale entry that
        duplicates on every subsequent ``load()`` and eventually crashes when
        the finalizer tries to detach a second time.
        """
        _comfy.model_management.free_memory(1e30, self.patcher.load_device)
        _comfy.model_management.soft_empty_cache()

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        t5_vec: torch.Tensor,
        clip_l_pooler: torch.Tensor,
        ref_latent_raw: Optional[torch.Tensor],
        noise: torch.Tensor,
        capture_indices: List[int],
        guidance: float = 2.5,
        seed: int = 42,
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        """Run the teacher sampler and capture x0 predictions at selected steps.

        Args:
            t5_vec: T5 text embeddings [B, seq, dim] — from musubi cache.
            clip_l_pooler: CLIP-L pooled output [B, dim] — from musubi cache.
            ref_latent_raw: Kontext reference latent in **raw (unscaled)** space
                [B, 16, H/8, W/8].  Caller must unscale cached latents before
                passing: ``raw = cached / 0.3611 + 0.1159``.
                ``None`` for unconditional / no-reference generation.
            noise: Random noise tensor [B, 16, H/8, W/8].
            capture_indices: Step indices at which to capture x0.
            guidance: Flux guidance value (default 2.5, Kontext's native).
            seed: RNG seed forwarded to comfy's sampler.

        Returns:
            (captured_x0, captured_sigmas) where:
              captured_x0:    {step_idx: x0_cpu_tensor} in *scaled* Kontext space
              captured_sigmas: {step_idx: sigma_float}
        """
        capture_set = set(capture_indices)
        captured_x0: Dict[int, torch.Tensor] = {}
        captured_sigmas: Dict[int, float] = {}

        # Build the sigma lookup for captured steps
        for idx in capture_indices:
            captured_sigmas[idx] = self.sigmas[idx].item()

        # -- Callback: 4 positional args (step, denoised_x0, x, total_steps) --
        # comfy/samplers.py wraps the k-diffusion dict callback into this form.
        # x0 is in scaled Kontext latent space (same as musubi's cached space).
        # Move to CPU immediately to avoid accumulating GPU tensors across the
        # whole chunk (fix 7).
        def capture_callback(step: int, denoised_x0: torch.Tensor, x: torch.Tensor, total_steps: int):
            if step in capture_set:
                captured_x0[step] = denoised_x0.detach().clone().to("cpu")

        # -- Conditioning (Flux format) --
        # positive = [[cross_attn_tensor, cond_dict]]
        # cross_attn_tensor is t5_vec (element 0); cond_dict carries pooled_output,
        # guidance, and optionally reference_latents.
        cond_dict = {
            "pooled_output": clip_l_pooler,
            "guidance": guidance,
        }
        if ref_latent_raw is not None:
            # comfy applies process_latent_in internally, so pass raw (unscaled)
            cond_dict["reference_latents"] = [ref_latent_raw]

        positive = [[t5_vec, cond_dict]]
        negative = []

        # latent_image must be zeros for txt2img-from-noise
        latent_image = torch.zeros_like(noise)

        _comfy.sample.sample(
            model=self.patcher,
            noise=noise,
            steps=self.num_steps,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=1.0,
            disable_noise=True,  # we supply our own noise
            callback=capture_callback,
            disable_pbar=True,
            seed=seed,
        )

        return captured_x0, captured_sigmas

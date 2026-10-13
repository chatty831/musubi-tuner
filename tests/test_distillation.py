"""Tests for FLUX.1 Kontext → FLUX.2 Klein 4B distillation utilities.

Covers pure-function unit tests (no GPU, no model weights required):
  - get_capture_indices
  - _unscale_kontext_latent / scaling round-trip
  - dual_arch_batch path rewriting and key normalisation
  - distillation_loss
"""

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.distillation.trajectory_capture import get_capture_indices
from musubi_tuner.distillation.loss import distillation_loss


# -----------------------------------------------------------------------
# get_capture_indices
# -----------------------------------------------------------------------

class TestGetCaptureIndices:
    def test_basic_uniform(self):
        result = get_capture_indices(15, 5)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == 14

    def test_boundary_single_capture(self):
        result = get_capture_indices(10, 1)
        assert result == [9]

    def test_captures_equal_steps(self):
        result = get_capture_indices(5, 5)
        assert result == [0, 1, 2, 3, 4]

    def test_captures_exceed_steps(self):
        result = get_capture_indices(3, 10)
        assert result == [0, 1, 2]

    def test_empty_steps(self):
        result = get_capture_indices(0, 5)
        assert result == []

    def test_four_captures_from_fifteen(self):
        """The default distillation config: 15 teacher steps, 4 student steps."""
        result = get_capture_indices(15, 4)
        assert len(result) == 4
        assert result[0] == 0
        assert result[-1] == 14
        # Verify monotonically increasing
        assert result == sorted(result)

    def test_indices_are_integers(self):
        result = get_capture_indices(15, 4)
        assert all(isinstance(i, int) for i in result)

    def test_two_captures(self):
        result = get_capture_indices(10, 2)
        assert result == [0, 9]


# -----------------------------------------------------------------------
# Kontext latent scaling
# -----------------------------------------------------------------------

class TestKontextScaling:
    def test_unscale_round_trip(self):
        """Unscaling then re-scaling should recover the original latent."""
        from musubi_tuner.flux_kontext_distill_train import _unscale_kontext_latent, _KONTEXT_SCALE, _KONTEXT_SHIFT

        z_scaled = torch.randn(2, 16, 32, 32)
        z_raw = _unscale_kontext_latent(z_scaled)
        z_rescaled = _KONTEXT_SCALE * (z_raw - _KONTEXT_SHIFT)
        assert torch.allclose(z_scaled, z_rescaled, atol=1e-5)

    def test_unscale_known_values(self):
        """Verify the formula z_raw = z_scaled / 0.3611 + 0.1159."""
        from musubi_tuner.flux_kontext_distill_train import _unscale_kontext_latent

        z_scaled = torch.tensor([0.3611])  # should give 1.0 + 0.1159 = 1.1159
        z_raw = _unscale_kontext_latent(z_scaled)
        assert z_raw.item() == pytest.approx(1.0 + 0.1159, rel=1e-4)


# -----------------------------------------------------------------------
# distillation_loss
# -----------------------------------------------------------------------

class TestDistillationLoss:
    def test_mse_zero_diff(self):
        x = torch.randn(2, 128, 8, 8)
        loss = distillation_loss(x, x, "mse")
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_l1_loss(self):
        pred = torch.zeros(2, 4)
        target = torch.ones(2, 4)
        loss = distillation_loss(pred, target, "l1")
        assert loss.item() == pytest.approx(1.0, rel=1e-5)

    def test_huber_loss(self):
        pred = torch.zeros(2, 4)
        target = torch.ones(2, 4)
        loss = distillation_loss(pred, target, "huber")
        assert loss.item() > 0

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            distillation_loss(torch.zeros(1), torch.zeros(1), "xyz")


# -----------------------------------------------------------------------
# dual_arch_batch: path rewriting
# -----------------------------------------------------------------------

class TestDualArchBatchPaths:
    def test_rewrite_latent_path(self):
        from musubi_tuner.distillation.dual_arch_batch import _rewrite_path, _STUDENT_LATENT_SUFFIX, _TEACHER_LATENT_SUFFIX
        path = r"C:\cache\img001_0512x0512_f2k4b.safetensors"
        result = _rewrite_path(path, _STUDENT_LATENT_SUFFIX, _TEACHER_LATENT_SUFFIX)
        assert result == r"C:\cache\img001_0512x0512_fk.safetensors"

    def test_rewrite_te_path(self):
        from musubi_tuner.distillation.dual_arch_batch import _rewrite_path, _STUDENT_TE_SUFFIX, _TEACHER_TE_SUFFIX
        path = "/data/cache/photo_f2k4b_te.safetensors"
        result = _rewrite_path(path, _STUDENT_TE_SUFFIX, _TEACHER_TE_SUFFIX)
        assert result == "/data/cache/photo_fk_te.safetensors"

    def test_rewrite_wrong_suffix_raises(self):
        from musubi_tuner.distillation.dual_arch_batch import _rewrite_path
        with pytest.raises(ValueError, match="Expected path ending"):
            _rewrite_path("bad_file.txt", "_f2k4b.safetensors", "_fk.safetensors")


# -----------------------------------------------------------------------
# dual_arch_batch: key normalisation
# -----------------------------------------------------------------------

class TestKeyNormalisation:
    def test_strip_dtype_suffix(self):
        from musubi_tuner.distillation.dual_arch_batch import _normalise_key
        assert _normalise_key("t5_vec_bfloat16") == "t5_vec"

    def test_strip_latent_shape(self):
        from musubi_tuner.distillation.dual_arch_batch import _normalise_key
        assert _normalise_key("latents_1x64x64_bfloat16") == "latents"

    def test_strip_control_latent_shape(self):
        from musubi_tuner.distillation.dual_arch_batch import _normalise_key
        assert _normalise_key("latents_control_1x64x64_bfloat16") == "latents_control"

    def test_clip_l_pooler(self):
        from musubi_tuner.distillation.dual_arch_batch import _normalise_key
        assert _normalise_key("clip_l_pooler_bfloat16") == "clip_l_pooler"

    def test_mask_key_preserved(self):
        from musubi_tuner.distillation.dual_arch_batch import _normalise_key
        assert _normalise_key("attention_mask") == "attention_mask"


# -----------------------------------------------------------------------
# dual_arch_batch: validation
# -----------------------------------------------------------------------

class TestValidateTeacherCaches:
    def _make_dummy_caches(self, tmpdir, basename="img001", size="0512x0512"):
        """Create minimal f2k4b and fk cache files in tmpdir."""
        dummy = {"x_bfloat16": torch.zeros(1)}
        f2k4b_lat = os.path.join(tmpdir, f"{basename}_{size}_f2k4b.safetensors")
        f2k4b_te = os.path.join(tmpdir, f"{basename}_f2k4b_te.safetensors")
        fk_lat = os.path.join(tmpdir, f"{basename}_{size}_fk.safetensors")
        fk_te = os.path.join(tmpdir, f"{basename}_fk_te.safetensors")
        for p in [f2k4b_lat, f2k4b_te, fk_lat, fk_te]:
            save_file(dummy, p)
        return f2k4b_lat, f2k4b_te

    def test_validation_passes_when_all_present(self):
        from musubi_tuner.distillation.dual_arch_batch import validate_teacher_caches
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            f2k4b_lat, f2k4b_te = self._make_dummy_caches(tmpdir)

            item = MagicMock()
            item.latent_cache_path = f2k4b_lat
            item.text_encoder_output_cache_path = f2k4b_te

            bm = MagicMock()
            bm.buckets = {(512, 512): [item]}

            validate_teacher_caches(bm)  # should not raise

    def test_validation_fails_when_fk_missing(self):
        from musubi_tuner.distillation.dual_arch_batch import validate_teacher_caches
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only f2k4b files, no fk files
            dummy = {"x_bfloat16": torch.zeros(1)}
            f2k4b_lat = os.path.join(tmpdir, "img_0512x0512_f2k4b.safetensors")
            f2k4b_te = os.path.join(tmpdir, "img_f2k4b_te.safetensors")
            save_file(dummy, f2k4b_lat)
            save_file(dummy, f2k4b_te)

            item = MagicMock()
            item.latent_cache_path = f2k4b_lat
            item.text_encoder_output_cache_path = f2k4b_te

            bm = MagicMock()
            bm.buckets = {(512, 512): [item]}

            with pytest.raises(FileNotFoundError, match="Missing.*teacher"):
                validate_teacher_caches(bm)


# -----------------------------------------------------------------------
# Parser args
# -----------------------------------------------------------------------

class TestParser:
    def test_distill_args_present(self):
        from musubi_tuner.flux_kontext_distill_train import flux_kontext_distill_setup_parser
        parser = flux_kontext_distill_setup_parser()

        # Verify required args exist by checking they raise when missing
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_distill_defaults(self):
        from musubi_tuner.flux_kontext_distill_train import flux_kontext_distill_setup_parser
        parser = flux_kontext_distill_setup_parser()
        args = parser.parse_args([
            "--distill_teacher_dit", "teacher.safetensors",
            "--distill_teacher_vae", "teacher_vae.safetensors",
            "--distill_student_vae", "student_vae.safetensors",
            # Also need the base trainer required args — use defaults where possible
            "--dit", "dit.safetensors",
            "--dataset_config", "data.toml",
            "--output_dir", "out",
        ])
        assert args.distill_guidance == 2.5
        assert args.distill_teacher_steps == 15
        assert args.distill_student_steps == 4
        assert args.distill_chunk_batches == 100
        assert args.distill_loss_type == "mse"
        assert args.distill_seed_mode == "per_item"
        assert args.distill_teacher_dtype == "bf16"


# -----------------------------------------------------------------------
# batch CPU/GPU movement helpers
# -----------------------------------------------------------------------

class TestBatchMovement:
    def test_batch_to_cpu(self):
        from musubi_tuner.flux_kontext_distill_train import _batch_to_cpu
        batch = {"t": torch.zeros(2, device="cpu"), "s": "hello", "n": None}
        result = _batch_to_cpu(batch)
        assert result["t"].device.type == "cpu"
        assert result["s"] == "hello"
        assert result["n"] is None

    def test_batch_to_device_cpu(self):
        from musubi_tuner.flux_kontext_distill_train import _batch_to_device
        batch = {"t": torch.zeros(2, device="cpu"), "x": 42}
        result = _batch_to_device(batch, "cpu")
        assert result["t"].device.type == "cpu"
        assert result["x"] == 42

    def test_round_trip_preserves_values(self):
        from musubi_tuner.flux_kontext_distill_train import _batch_to_cpu, _batch_to_device
        original = torch.randn(3, 4)
        batch = {"t": original.clone()}
        batch = _batch_to_cpu(batch)
        batch = _batch_to_device(batch, "cpu")
        assert torch.allclose(batch["t"], original)


# -----------------------------------------------------------------------
# dual_arch_batch: __class__ reassignment (dunder resolution fix)
# -----------------------------------------------------------------------

class TestDistillBatchManagerSubclass:
    def test_class_reassignment(self):
        """After wrap_batch_manager_for_distillation, type(bm).__getitem__
        must resolve to the distill subclass, not the base BucketBatchManager."""
        from musubi_tuner.distillation.dual_arch_batch import (
            wrap_batch_manager_for_distillation,
            _DistillBucketBatchManager,
        )
        from musubi_tuner.dataset.bucket import BucketBatchManager

        bm = BucketBatchManager({}, batch_size=1)
        assert type(bm) is BucketBatchManager
        wrap_batch_manager_for_distillation(bm)
        assert type(bm) is _DistillBucketBatchManager
        # isinstance still works (it's a subclass)
        assert isinstance(bm, BucketBatchManager)

    def test_subclass_is_pickleable(self):
        """The subclass must survive pickle for Windows multiprocessing."""
        import pickle
        from musubi_tuner.distillation.dual_arch_batch import (
            wrap_batch_manager_for_distillation,
        )
        from musubi_tuner.dataset.bucket import BucketBatchManager

        bm = BucketBatchManager({}, batch_size=1)
        wrap_batch_manager_for_distillation(bm)
        # Should not raise
        data = pickle.dumps(bm)
        bm2 = pickle.loads(data)
        assert type(bm2).__name__ == "_DistillBucketBatchManager"


# -----------------------------------------------------------------------
# dry_run: mock shapes and activate/deactivate
# -----------------------------------------------------------------------

class TestDryRunTeacherPipeline:
    def test_generate_trajectory_shapes(self):
        from musubi_tuner.distillation.dry_run import DryRunTeacherPipeline

        pipe = DryRunTeacherPipeline(num_steps=15)
        noise = torch.randn(2, 16, 64, 64)
        caps = [0, 5, 10, 14]

        x0s, sigmas = pipe.generate_trajectory(
            t5_vec=torch.zeros(2, 512, 4096),
            clip_l_pooler=torch.zeros(2, 768),
            ref_latent_raw=torch.zeros(2, 16, 64, 64),
            noise=noise,
            capture_indices=caps,
        )

        assert len(x0s) == 4
        assert len(sigmas) == 4
        for idx in caps:
            assert x0s[idx].shape == (2, 16, 64, 64)
            assert x0s[idx].device.type == "cpu"
            assert isinstance(sigmas[idx], float)

    def test_sigma_schedule(self):
        from musubi_tuner.distillation.dry_run import DryRunTeacherPipeline

        pipe = DryRunTeacherPipeline(num_steps=10)
        assert pipe.sigmas[0].item() == pytest.approx(1.0, abs=1e-5)
        assert pipe.sigmas[-1].item() == pytest.approx(0.0, abs=1e-5)
        assert len(pipe.sigmas) == 11  # num_steps + 1


class TestDryRunMockVAE:
    def test_kontext_decode_shapes(self):
        from musubi_tuner.distillation.dry_run import _mock_kontext_decode

        z = torch.zeros(2, 16, 32, 32)
        out = _mock_kontext_decode(None, z)
        assert out.shape == (2, 3, 256, 256)  # 32*8 = 256

    def test_flux2_encode_shapes(self):
        from musubi_tuner.distillation.dry_run import _mock_flux2_encode

        pixels = torch.zeros(2, 3, 512, 512)
        out = _mock_flux2_encode(None, pixels)
        assert out.shape == (2, 128, 32, 32)  # 512/16 = 32


class TestDryRunCallDit:
    def test_returns_correct_shapes(self):
        from musubi_tuner.distillation.dry_run import _dry_run_call_dit
        from unittest.mock import MagicMock

        latents = torch.randn(1, 128, 32, 32)
        noise = torch.randn(1, 128, 32, 32)
        noisy = torch.randn(1, 128, 32, 32)
        timesteps = torch.tensor([500.0])

        mock_self = MagicMock()
        result = _dry_run_call_dit(
            mock_self, None, None, None, latents, {},
            noise, noisy, timesteps, torch.bfloat16,
        )

        assert result.pred.shape == (1, 128, 32, 32)
        assert result.target.shape == (1, 128, 32, 32)
        # target should be noise - latents
        assert torch.allclose(result.target, noise - latents)


class TestDryRunActivateDeactivate:
    def test_patches_applied_and_removed(self):
        from musubi_tuner.distillation.dry_run import activate_dry_run, deactivate_dry_run, _patches
        from argparse import Namespace

        args = Namespace(
            max_train_steps=1000,
            max_train_epochs=None,
            distill_chunk_batches=100,
            save_every_n_steps=500,
            sample_every_n_steps=500,
            sample_prompts="foo.json",
            max_data_loader_n_workers=4,
            persistent_data_loader_workers=True,
            save_state=True,
            save_state_on_train_end=True,
            compile=True,
            gradient_checkpointing=True,
            blocks_to_swap=0,
            distill_teacher_steps=15,
            distill_student_steps=4,
        )

        activate_dry_run(args)
        try:
            # Args should be forced
            assert args.max_train_steps == 2
            assert args.distill_chunk_batches == 2
            assert args.sample_prompts is None
            assert args.compile is False
            # Patches should be active
            assert len(_patches) > 0
        finally:
            deactivate_dry_run()

        # Patches should be removed
        assert len(_patches) == 0


class TestDryRunDataset:
    def test_synthetic_batch_keys(self):
        from musubi_tuner.distillation.dry_run import _DryRunDatasetGroup

        ds = _DryRunDatasetGroup(batch_size=2, height=512, width=512, num_batches=3)
        batch = ds[0]

        # Student keys
        assert "latents" in batch
        assert batch["latents"].shape == (2, 128, 32, 32)
        assert "ctx_vec" in batch

        # Teacher keys
        assert "teacher_t5_vec" in batch
        assert batch["teacher_t5_vec"].shape == (2, 512, 4096)
        assert "teacher_clip_l_pooler" in batch
        assert batch["teacher_clip_l_pooler"].shape == (2, 768)

"""Cross-architecture batch joining for FLUX.1 Kontext -> FLUX.2 distillation.

After ``prepare_for_training()`` builds a ``BucketBatchManager`` for the
student (f2k4b) architecture, call ``wrap_batch_manager_for_distillation``
to augment each batch with the corresponding Kontext (fk) teacher data.

For every item in a batch the wrapper:
  1. Derives the ``fk`` cache paths from the ``f2k4b`` paths by suffix rewrite.
  2. Loads the ``fk`` latent and TE caches.
  3. Normalises keys identically to ``BucketBatchManager.__getitem__``.
  4. Merges them into the batch dict with a ``teacher_`` prefix
     (``teacher_latents``, ``teacher_latents_control``, ``teacher_t5_vec``,
     ``teacher_clip_l_pooler``).

The wrapper uses ``__class__`` reassignment to a module-level subclass so
that Python's dunder resolution (``type(obj).__getitem__``) finds the
override, and the object remains pickleable for multi-worker dataloaders
on Windows.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_2_KLEIN_4B, ARCHITECTURE_FLUX_KONTEXT
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.utils.model_utils import remove_dtype_suffix

logger = logging.getLogger(__name__)

# Suffix patterns used in cache filenames
_STUDENT_LATENT_SUFFIX = f"_{ARCHITECTURE_FLUX_2_KLEIN_4B}.safetensors"
_TEACHER_LATENT_SUFFIX = f"_{ARCHITECTURE_FLUX_KONTEXT}.safetensors"
_STUDENT_TE_SUFFIX = f"_{ARCHITECTURE_FLUX_2_KLEIN_4B}_te.safetensors"
_TEACHER_TE_SUFFIX = f"_{ARCHITECTURE_FLUX_KONTEXT}_te.safetensors"


def _rewrite_path(path: str, old_suffix: str, new_suffix: str) -> str:
    """Replace the architecture suffix in a cache file path."""
    if not path.endswith(old_suffix):
        raise ValueError(f"Expected path ending with '{old_suffix}', got: {path}")
    return path[: -len(old_suffix)] + new_suffix


def _normalise_key(raw_key: str) -> str:
    """Normalise a safetensors key identically to BucketBatchManager.__getitem__."""
    key = raw_key
    is_varlen = key.startswith("varlen_")
    if is_varlen:
        key = key.replace("varlen_", "", 1)
    if key.endswith("_mask"):
        return key
    key = remove_dtype_suffix(key)
    if key.startswith("latents_"):
        key = key.rsplit("_", 1)[0]  # strip FxHxW
    return key


def validate_teacher_caches(batch_manager: BucketBatchManager) -> None:
    """Check that every f2k4b item has matching fk latent + TE cache files.

    Raises ``FileNotFoundError`` listing all missing files.
    """
    missing = []
    seen_paths: set[str] = set()

    for bucket_reso, bucket in batch_manager.buckets.items():
        for item_info in bucket:
            lat_path = item_info.latent_cache_path
            if lat_path in seen_paths:
                continue
            seen_paths.add(lat_path)

            teacher_lat = _rewrite_path(lat_path, _STUDENT_LATENT_SUFFIX, _TEACHER_LATENT_SUFFIX)
            teacher_te = _rewrite_path(
                item_info.text_encoder_output_cache_path, _STUDENT_TE_SUFFIX, _TEACHER_TE_SUFFIX
            )
            if not os.path.exists(teacher_lat):
                missing.append(teacher_lat)
            if not os.path.exists(teacher_te):
                missing.append(teacher_te)

    if missing:
        msg = f"Missing {len(missing)} teacher (fk) cache file(s):\n" + "\n".join(f"  {p}" for p in missing[:20])
        if len(missing) > 20:
            msg += f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(msg)

    logger.info(f"Teacher cache validation passed: {len(seen_paths)} items OK")


def _load_teacher_tensors(item_info: Any) -> dict[str, torch.Tensor]:
    """Load the fk caches for a single item and return normalised key -> tensor mapping."""
    teacher_lat_path = _rewrite_path(item_info.latent_cache_path, _STUDENT_LATENT_SUFFIX, _TEACHER_LATENT_SUFFIX)
    teacher_te_path = _rewrite_path(
        item_info.text_encoder_output_cache_path, _STUDENT_TE_SUFFIX, _TEACHER_TE_SUFFIX
    )

    sd_lat = load_file(teacher_lat_path)
    sd_te = load_file(teacher_te_path)
    sd = {**sd_lat, **sd_te}

    result: dict[str, torch.Tensor] = {}
    for raw_key, tensor in sd.items():
        key = _normalise_key(raw_key)
        result[key] = tensor
    return result


# ---------------------------------------------------------------------------
# Module-level subclass — pickleable, visible to type(obj).__getitem__
# ---------------------------------------------------------------------------

class _DistillBucketBatchManager(BucketBatchManager):
    """BucketBatchManager subclass that merges teacher (fk) data into each batch.

    Defined at module scope so it is importable and pickleable (required by
    Windows multiprocessing / ``--max_data_loader_n_workers > 0``).
    """

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))

        teacher_batch: dict[str, list[torch.Tensor]] = {}
        for item_info in bucket[start:end]:
            teacher_sd = _load_teacher_tensors(item_info)
            for key, tensor in teacher_sd.items():
                teacher_key = f"teacher_{key}"
                if teacher_key not in teacher_batch:
                    teacher_batch[teacher_key] = []
                teacher_batch[teacher_key].append(tensor)

        for key, tensors in teacher_batch.items():
            batch[key] = torch.stack(tensors)

        return batch


def wrap_batch_manager_for_distillation(batch_manager: BucketBatchManager) -> None:
    """Reassign ``batch_manager.__class__`` to the distillation subclass.

    This makes ``type(batch_manager).__getitem__`` resolve to the override
    (Python looks up dunder methods on the type, not the instance).  Because
    ``_DistillBucketBatchManager`` is a module-level class the object stays
    pickleable for multi-worker dataloaders.
    """
    batch_manager.__class__ = _DistillBucketBatchManager

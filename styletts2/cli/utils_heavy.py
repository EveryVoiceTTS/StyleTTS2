"""Heavy synthesis helpers for StyleTTS2 — imported lazily to keep CLI startup fast."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Sequence

import lightning as L
import torch
import torchaudio
from everyvoice import logger
from everyvoice.base_cli.prediction_writing_callback import (
    BasePredictionWritingCallback,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import (
    truncate_basename,
)
from everyvoice.utils import slugify
from torch.utils.data import DataLoader, Dataset


def build_text_entries(
    texts: list[str],
    reference_path: str,
    speaker: str,
    language: str,
    diffusion_steps: int,
    embedding_scale: float,
    acoustic_blend: float,
    prosody_blend: float,
) -> list[dict]:
    """Build one synthesis entry per `--text` string, all sharing the same
    reference/speaker/language."""
    return [
        {
            "raw_text": t,
            "basename": truncate_basename(slugify(t)),
            "speaker": speaker,
            "language": language,
            "reference_path": reference_path,
            "diffusion_steps": diffusion_steps,
            "embedding_scale": embedding_scale,
            "acoustic_blend": acoustic_blend,
            "prosody_blend": prosody_blend,
        }
        for t in texts
    ]


def build_filelist_entries(
    rows: Iterable[dict],
    default_reference_path: Optional[str],
    default_speaker: str,
    default_language: str,
    diffusion_steps: int,
    embedding_scale: float,
    acoustic_blend: float,
    prosody_blend: float,
) -> list[dict]:
    """Build one synthesis entry per filelist row.

    Each row may supply its own 'basename', 'speaker', 'language', and
    'reference'/'reference_path' columns; any that are absent fall back to
    the corresponding default. Raises ValueError if a row has no
    'characters'/'phones' column, or no reference path can be resolved.
    """
    entries = []
    for d in rows:
        raw_text = d.get("characters") or d.get("phones")
        if not raw_text:
            raise ValueError(
                f"Filelist row is missing a 'characters' or 'phones' column: {d}"
            )
        reference_path = (
            d.get("reference_path") or d.get("reference") or default_reference_path
        )
        if not reference_path:
            raise ValueError(
                "Missing --reference option, and this filelist row has no"
                f" 'reference'/'reference_path' column: {d}"
            )
        entries.append(
            {
                "raw_text": raw_text,
                "basename": d.get("basename", truncate_basename(slugify(raw_text))),
                "speaker": d.get("speaker", default_speaker),
                "language": d.get("language", default_language),
                "reference_path": reference_path,
                "diffusion_steps": diffusion_steps,
                "embedding_scale": embedding_scale,
                "acoustic_blend": acoustic_blend,
                "prosody_blend": prosody_blend,
            }
        )
    return entries


def _synthesis_collate_fn(batch):
    assert len(batch) == 1, "StyleTTS2 synthesis requires batch_size=1"
    return batch[0]


class StyleTTS2SynthesisDataset(Dataset):
    def __init__(self, entries: list[dict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


class StyleTTS2SynthesisDataModule(L.LightningDataModule):
    def __init__(self, entries: list[dict]):
        super().__init__()
        self.entries = entries

    def predict_dataloader(self):
        return DataLoader(
            StyleTTS2SynthesisDataset(self.entries),
            batch_size=1,
            collate_fn=_synthesis_collate_fn,
            shuffle=False,
            num_workers=0,
        )


class StyleTTS2PredictionWritingWavCallback(BasePredictionWritingCallback):
    def __init__(
        self, save_dir: Path, global_step: int, simple_filenames: bool = False
    ):
        super().__init__(
            save_dir=save_dir / "wav",
            file_extension="pred.wav",
            global_step=global_step,
            include_global_step_in_filename=True,
            simple_filenames=simple_filenames,
        )
        self.last_file_written: Optional[str] = None

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        _trainer,
        _pl_module,
        outputs,
        batch,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ):
        if outputs is None:
            return
        wav_tensor = torch.from_numpy(outputs["wav"]).unsqueeze(0)
        filename = self.get_filename(
            outputs["basename"], outputs["speaker"], outputs["language"]
        )
        torchaudio.save(
            filename,
            wav_tensor,
            outputs["sample_rate"],
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
        )
        self.last_file_written = filename
        logger.info(f"Saved WAV: {filename}")


def get_styletts2_synthesis_output_callbacks(
    output_type: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    global_step: int,
    sample_rate: int,
    simple_filenames: bool = False,
) -> dict[SynthesizeOutputFormats, BasePredictionWritingCallback]:
    """Build the set of synthesis callbacks for the requested output formats. Only supports wav for now."""
    callbacks: dict[SynthesizeOutputFormats, BasePredictionWritingCallback] = {}
    if SynthesizeOutputFormats.wav in output_type:
        callbacks[SynthesizeOutputFormats.wav] = StyleTTS2PredictionWritingWavCallback(
            output_dir, global_step, simple_filenames=simple_filenames
        )
    return callbacks

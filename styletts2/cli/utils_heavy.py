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
from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.prediction_writing_callback import (
    resolve_chunked_basename,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.utils import (
    truncate_basename,
)
from everyvoice.text.textsplit import chunk_text
from everyvoice.utils import slugify
from torch.utils.data import DataLoader, Dataset


def get_styletts2_text_split_params(
    module, language: Optional[str], text_representation: DatasetTextRepresentation
) -> "tuple[bool, tuple[int, int, str, str]]":
    """Calculate text-chunking parameters for StyleTTS2, mirroring FastSpeech2's
    ``get_text_split_params``.

    StyleTTS2 has no per-model length statistics (unlike FastSpeech2's ``Stats``,
    saved at training time), so ``desired_length``/``max_length`` always fall
    back to the same hardcoded defaults FastSpeech2 itself uses for older
    models that lack stats.
    """
    text_config = module.config["ev_config"].text
    split_text: bool = text_config.split_text

    strong_boundaries = ""
    weak_boundaries = ""
    desired_length = 100
    max_length = 200

    if split_text:
        try:
            effective_language = language or ""
            strong_boundaries = text_config.boundaries[effective_language].strong
            weak_boundaries = text_config.boundaries[effective_language].weak
        except KeyError:
            logger.warning(
                f"Boundaries for language '{language}' could not be found in "
                "TextConfig. Chunking will not be performed."
            )

    return split_text, (
        int(desired_length),
        int(max_length),
        strong_boundaries,
        weak_boundaries,
    )


def build_text_entries(
    texts: list[str],
    reference_path: str,
    speaker: str,
    language: Optional[str],
    diffusion_steps: int,
    embedding_scale: float,
    acoustic_blend: float,
    prosody_blend: float,
    text_representation: DatasetTextRepresentation = DatasetTextRepresentation.characters,
    split_text: bool = False,
    split_params: tuple[int, int, str, str] = (100, 200, "", ""),
) -> list[dict]:
    """Build one synthesis entry per `--text` string, all sharing the same
    reference/speaker/language. When `split_text` is set, long strings are
    split into multiple chunk entries (via `chunk_text`) sharing the same
    basename, flagged with `is_last_input_chunk` for reassembly at write time."""
    entries = []
    for t in texts:
        chunks = chunk_text(t, *split_params) if split_text else [t]
        basename = truncate_basename(slugify(t))
        for i, chunk in enumerate(chunks):
            entries.append(
                {
                    "raw_text": chunk,
                    "basename": basename,
                    "speaker": speaker,
                    "language": language,
                    "reference_path": reference_path,
                    "diffusion_steps": diffusion_steps,
                    "embedding_scale": embedding_scale,
                    "acoustic_blend": acoustic_blend,
                    "prosody_blend": prosody_blend,
                    "text_representation": text_representation,
                    "is_last_input_chunk": i == len(chunks) - 1,
                }
            )
    return entries


def build_filelist_entries(
    rows: Iterable[dict],
    default_reference_path: Optional[str],
    default_speaker: str,
    default_language: Optional[str],
    diffusion_steps: int,
    embedding_scale: float,
    acoustic_blend: float,
    prosody_blend: float,
    split_text: bool = False,
    split_params: tuple[int, int, str, str] = (100, 200, "", ""),
) -> list[dict]:
    """Build one synthesis entry per filelist row (or per chunk of it, when
    `split_text` is set).

    Each row may supply its own 'basename', 'speaker', 'language', and
    'reference'/'reference_path' columns; any that are absent fall back to
    the corresponding default. The row's text representation is inferred
    from which of 'characters'/'phones' is present (matching precedence),
    so it doesn't need to be passed in. Raises ValueError if a row has
    neither column, or no reference path can be resolved.
    """
    entries = []
    for d in rows:
        if d.get("characters"):
            raw_text = d["characters"]
            text_representation = DatasetTextRepresentation.characters
        elif d.get("phones"):
            raw_text = d["phones"]
            text_representation = DatasetTextRepresentation.ipa_phones
        else:
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
        basename = d.get("basename", truncate_basename(slugify(raw_text)))
        speaker = d.get("speaker", default_speaker)
        language = d.get("language", default_language)
        chunks = chunk_text(raw_text, *split_params) if split_text else [raw_text]
        for i, chunk in enumerate(chunks):
            entries.append(
                {
                    "raw_text": chunk,
                    "basename": basename,
                    "speaker": speaker,
                    "language": language,
                    "reference_path": reference_path,
                    "diffusion_steps": diffusion_steps,
                    "embedding_scale": embedding_scale,
                    "acoustic_blend": acoustic_blend,
                    "prosody_blend": prosody_blend,
                    "text_representation": text_representation,
                    "is_last_input_chunk": i == len(chunks) - 1,
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
        self.full_wav = torch.tensor(())  # Accumulates full wav before saving file
        self.full_text: str = ""  # Accumulates full text before saving file
        self.chunk_basenames: list[str] = []  # Accumulates per-chunk basenames

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

        # Concatenate the current chunk onto the accumulated wav/text for this
        # utterance (a no-op when the input wasn't split into chunks, since
        # is_last_input_chunk is True on every single-chunk entry).
        self.full_wav = torch.cat((self.full_wav, wav_tensor), -1)
        self.full_text += outputs["raw_text"]
        self.chunk_basenames.append(outputs["basename"])

        if not outputs.get("is_last_input_chunk", True):
            return

        basename = resolve_chunked_basename(self.chunk_basenames, self.full_text)
        filename = self.get_filename(basename, outputs["speaker"], outputs["language"])
        torchaudio.save(
            filename,
            self.full_wav,
            outputs["sample_rate"],
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
        )
        self.last_file_written = filename
        logger.info(f"Saved WAV: {filename}")

        # Reset the accumulator variables
        self.full_wav = torch.tensor(())
        self.full_text = ""
        self.chunk_basenames = []


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

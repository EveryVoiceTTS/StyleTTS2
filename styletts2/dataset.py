from __future__ import annotations

import os.path as osp
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio  # noqa: F401 (kept for downstream imports)
from loguru import logger
from torch.utils.data import DataLoader

from .text_utils import TextCleaner
from .utils import MEL_MEAN, MEL_STD, make_mel_transform

if TYPE_CHECKING:
    from everyvoice.config.text_config import TextConfig

np.random.seed(1)
random.seed(1)


class FilePathDataset(torch.utils.data.Dataset):
    """StyleTTS2 dataset supporting two data-source modes:

    **Original mode** (``preprocessed_dir=None``):
        ``data_list`` is a list of raw strings from the native StyleTTS2
        pipe-separated filelist: ``wave_path|text|speaker_id``.
        Audio is resolved relative to ``root_path``.

    **EveryVoice mode** (``preprocessed_dir`` is set):
        ``data_list`` is a list of dicts loaded from an EveryVoice PSV
        filelist (keys: ``basename``, ``speaker``, ``language``,
        ``characters``, ``character_tokens``, …).  Audio is resolved from
        the EveryVoice-preprocessed directory using the same naming
        convention as FastSpeech2:
        ``{preprocessed_dir}/audio/{basename}--{speaker}--{language}--audio-{sr}.wav``
    """

    def __init__(
        self,
        data_list,
        root_path,
        config,
        preprocessed_dir=None,
        output_sampling_rate=None,
        speaker2id=None,
        ev_text_config: TextConfig | None = None,
        pretrained_symbols: list[str] | None = None,
        data_augmentation=False,
        validation=False,
        OOD_data="data/OOD_texts.txt",
        min_length=50,
        ood_data_paths: "dict[str, Any] | None" = None,
        ood_val_list: "list[dict] | None" = None,
    ):
        pp = config["preprocess_params"]
        self.sr = pp.get("sr", 24000)
        self.sep = "--"

        # EveryVoice preprocessed-directory mode
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.output_sampling_rate = output_sampling_rate or self.sr
        self.speaker2id = speaker2id  # {speaker_name: int_id} — required in EV mode

        if self.preprocessed_dir is not None:
            # data_list items are dicts from the EveryVoice PSV filelist loader
            self.data_list = data_list
            self.df = pd.DataFrame(
                [
                    {"basename": d["basename"], "speaker": d["speaker"]}
                    for d in data_list
                ]
            )
        else:
            # Original format: pipe-separated strings
            _data_list = [dl.strip().split("|") for dl in data_list]
            self.data_list = [
                data if len(data) == 3 else (*data, 0) for data in _data_list
            ]
            self.df = pd.DataFrame(self.data_list)

        # EveryVoice text encoder — only built when both config and symbols are provided.
        if ev_text_config is not None and pretrained_symbols is not None:
            from .ev_config.text import EVStyleTTS2TextEncoder

            self._ev_encoder: EVStyleTTS2TextEncoder | None = EVStyleTTS2TextEncoder(
                ev_text_config, pretrained_symbols
            )
            target_repr = config["data_params"].get(
                "target_text_representation", "characters"
            )
            self._token_column = (
                "phone_tokens" if target_repr == "phones" else "character_tokens"
            )
        else:
            self._ev_encoder = None
            self._token_column = "character_tokens"

        self.text_cleaner = TextCleaner()
        self.to_mel = make_mel_transform(config)
        self.mel_mean, self.mel_std = MEL_MEAN, MEL_STD
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = pp.get("max_mel_length", 192)
        self.silence_pad_samples = pp.get("silence_pad_samples", 5000)
        self.min_length = min_length

        # OOD texts: EV mode uses per-language PSV files; original mode uses
        # the legacy plain-text OOD_data path.
        if self.preprocessed_dir is not None:
            self.ood_texts: dict[str, list[tuple[str, list[int]]]] = {}
            self._load_ood_ev(ood_data_paths, ood_val_list)
            # Flat pool used when the current utterance's language has no OOD entry.
            self._ood_fallback_pool: list[tuple[str, list[int]]] = [
                item for items in self.ood_texts.values() for item in items
            ]
        else:
            with open(OOD_data, "r", encoding="utf-8") as f:
                tl = f.readlines()
            idx = 1 if ".wav" in tl[0].split("|")[0] else 0
            self.ptexts: list[str] = [t.split("|")[idx] for t in tl]

        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    # ------------------------------------------------------------------
    # OOD loading helper (EV mode)
    # ------------------------------------------------------------------

    def _load_ood_ev(
        self,
        ood_data_paths: "dict[str, Any] | None",
        ood_val_list: "list[dict] | None",
    ) -> None:
        """Populate self.ood_texts from per-language PSV paths or a val-fallback list."""
        from everyvoice.utils import generic_psv_filelist_reader

        # Collect raw PSV rows per language from whichever source was provided.
        source_rows: dict[str, list[dict]] = {}
        if ood_data_paths:
            for lang, path in ood_data_paths.items():
                source_rows[lang] = generic_psv_filelist_reader(path)
        elif ood_val_list is not None:
            for item in ood_val_list:
                lang = item.get("language", "und")
                source_rows.setdefault(lang, []).append(item)

        warned_langs: set[str] = set()
        for lang, rows in source_rows.items():
            lang_texts: list[tuple[str, list[int]]] = []
            for row in rows:
                token_str = row.get(self._token_column, "")
                if not token_str:
                    continue
                raw_text = row.get("characters", row.get("phones", ""))
                if self._ev_encoder is not None:
                    indices = self._ev_encoder.encode_token_sequence(token_str)
                else:
                    indices = self.text_cleaner(raw_text)
                if indices:
                    lang_texts.append((raw_text, indices))
            if lang_texts:
                self.ood_texts[lang] = lang_texts
            elif lang not in warned_langs:
                warned_langs.add(lang)
                logger.warning(
                    "OOD data for language '%s': column '%s' is absent or empty in all rows — "
                    "this language will fall back to the combined OOD pool.",
                    lang,
                    self._token_column,
                )

    # ------------------------------------------------------------------
    # EveryVoice-mode helpers
    # ------------------------------------------------------------------

    def _load_file(self, bn, spk, lang, dn, fn):
        """Return the Path for a file in the EveryVoice preprocessed directory.

        Mirrors fs2/dataset.py::FastSpeechDataset._load_file, but returns a
        Path rather than a loaded tensor because wav files are read with
        soundfile rather than torch.load.
        """
        return self.preprocessed_dir / dn / self.sep.join([bn, spk, lang, fn])

    def _load_tensor_ev(self, item):
        """Load wave + text tensor + speaker id from an EveryVoice filelist dict."""
        bn, spk, lang = item["basename"], item["speaker"], item["language"]
        speaker_id = self.speaker2id[spk] if self.speaker2id is not None else 0

        wav_path = self._load_file(
            bn, spk, lang, "audio", f"audio-{self.output_sampling_rate}.wav"
        )
        wave, sr = sf.read(wav_path)
        if wave.ndim == 2:
            wave = wave[:, 0]
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)

        pad = np.zeros([self.silence_pad_samples])
        wave = np.concatenate([pad, wave, pad], axis=0)

        if self._ev_encoder is not None:
            # Use the EveryVoice-preprocessed token column so that normalisation
            # and (optionally) G2P have already been applied by everyvoice preprocess.
            token_str = item.get(self._token_column, "")
            indices = self._ev_encoder.encode_token_sequence(token_str)
        else:
            # Fallback: raw character string through StyleTTS2's native TextCleaner.
            raw_text = item.get("characters", "")
            indices = self.text_cleaner(raw_text)

        indices.insert(0, 0)  # prepend StyleTTS2 pad/boundary symbol ($)
        indices.append(0)
        text = torch.LongTensor(indices)

        return wave, text, speaker_id

    def _load_data_ev(self, item):
        wave, text_tensor, speaker_id = self._load_tensor_ev(item)
        mel_tensor = self._preprocess(wave).squeeze()
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[
                :, random_start : random_start + self.max_mel_length
            ]
        return mel_tensor, speaker_id

    # ------------------------------------------------------------------
    # Original-mode helpers (unchanged)
    # ------------------------------------------------------------------

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
            print(wave_path, sr)

        pad = np.zeros([self.silence_pad_samples])
        wave = np.concatenate([pad, wave, pad], axis=0)

        text = self.text_cleaner(text)
        text.insert(0, 0)
        text.append(0)
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        return (
            torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mel_mean
        ) / self.mel_std

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = self._preprocess(wave).squeeze()
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[
                :, random_start : random_start + self.max_mel_length
            ]
        return mel_tensor, speaker_id

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.preprocessed_dir is not None:
            wave, text_tensor, speaker_id = self._load_tensor_ev(data)
            # Reference sample: another utterance from the same speaker
            speaker = data["speaker"]
            ref_rows = self.df[self.df["speaker"] == speaker]
            ref_item = self.data_list[ref_rows.sample(n=1).index[0]]
            ref_mel_tensor, ref_label = self._load_data_ev(ref_item)
            path = data["basename"]
        else:
            wave, text_tensor, speaker_id = self._load_tensor(data)
            ref_data = (
                (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            )
            ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
            path = data[0]

        mel_tensor = self._preprocess(wave).squeeze()
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        # OOD reference text — language-keyed in EV mode, plain-text pool in original mode.
        if self.preprocessed_dir is not None:
            lang = data.get("language", "und") if isinstance(data, dict) else "und"
            pool = self.ood_texts.get(lang) or self._ood_fallback_pool
            if not pool:
                # No OOD data at all — use training text as a last resort.
                ref_text = text_tensor
            else:
                ps_text = ""
                while len(ps_text) < self.min_length:
                    rand_idx = np.random.randint(0, len(pool))
                    ps_text, indices = pool[rand_idx % len(pool)]
                ref_text = torch.LongTensor([0] + list(indices) + [0])
        else:
            ps_text = ""
            while len(ps_text) < self.min_length:
                rand_idx = np.random.randint(0, len(self.ptexts) - 1)
                ps_text = self.ptexts[rand_idx]
                indices = self.text_cleaner(ps_text)
            ref_text = torch.LongTensor([0] + list(indices) + [0])

        return (
            speaker_id,
            acoustic_feature,
            text_tensor,
            ref_text,
            ref_mel_tensor,
            ref_label,
            path,
            wave,
        )


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, max_mel_length=192, return_wave=False):
        self.text_pad_index = 0
        self.max_mel_length = max_mel_length
        self.return_wave = return_wave

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ["" for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        for bid, (
            label,
            mel,
            text,
            ref_text,
            ref_mel,
            ref_label,
            path,
            wave,
        ) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            ref_labels[bid] = ref_label
            waves[bid] = wave

        return (
            waves,
            texts,
            input_lengths,
            ref_texts,
            ref_lengths,
            mels,
            output_lengths,
            ref_mels,
            paths,
        )


def build_dataloader(
    path_list,
    root_path,
    config,
    preprocessed_dir=None,
    output_sampling_rate=None,
    speaker2id=None,
    ev_text_config=None,
    pretrained_symbols=None,
    validation=False,
    OOD_data="data/OOD_texts.txt",
    min_length=50,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config={},
    dataset_config={},
    ood_data_paths=None,
    ood_val_list=None,
):
    max_mel_length = config["preprocess_params"].get("max_mel_length", 192)
    dataset = FilePathDataset(
        path_list,
        root_path,
        config,
        preprocessed_dir=preprocessed_dir,
        output_sampling_rate=output_sampling_rate,
        speaker2id=speaker2id,
        ev_text_config=ev_text_config,
        pretrained_symbols=pretrained_symbols,
        OOD_data=OOD_data,
        min_length=min_length,
        validation=validation,
        ood_data_paths=ood_data_paths,
        ood_val_list=ood_val_list,
        **dataset_config,
    )
    collate_fn = Collater(max_mel_length=max_mel_length, **collate_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )
    return data_loader

"""Tests for OOD text sampling in FilePathDataset.__getitem__."""

import os
import signal
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from styletts2.dataset import FilePathDataset
from styletts2.text_utils import TextCleaner

_SPACE_IDX = TextCleaner()(" ")[0]  # index of ' ' in the StyleTTS2 symbol table

_OOD_PSV = Path(__file__).parent / "test_validation_ood.psv"


@contextmanager
def _time_limit(seconds=5):
    if os.name != "nt":
        # Linux or macOS; more efficient, interrupts a test exceeding its time

        def _handler(signum, frame):
            raise TimeoutError(f"OOD sampling did not terminate within {seconds}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback, where SIGALRM is not implemented; has to wait until the test
        # is done, we can only detect that it went overtime, not interrupt it.
        timed_out = False

        def _windows_handler():
            nonlocal timed_out
            timed_out = True

        timer = threading.Timer(seconds, _windows_handler)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            if timed_out:
                raise TimeoutError(f"OOD sampling did not terminate within {seconds}s")


def _dummy_tensors():
    wave = np.zeros(1000)
    text = torch.LongTensor([0, 1, 2, 0])
    mel = torch.zeros(80, 100)
    return wave, text, mel


class TestOODSamplingEVMode(unittest.TestCase):
    """OOD sampling path inside __getitem__ when preprocessed_dir is set."""

    def _make_dataset(self, pool, min_length):
        ds = object.__new__(FilePathDataset)
        ds.preprocessed_dir = MagicMock()
        ds.OOD_min_length = min_length
        ds.text_cleaner = TextCleaner()
        ds.ood_texts = {"und": pool}
        ds._ood_fallback_pool = pool
        ds.lang2id = None
        ds.data_list = [{"basename": "b0", "speaker": "spk0", "language": "und"}]
        ds.df = pd.DataFrame([{"basename": "b0", "speaker": "spk0"}])
        return ds

    def _call_getitem(self, ds):
        wave, text, mel = _dummy_tensors()
        with (
            patch.object(ds, "_load_tensor_ev", return_value=(wave, text, 0, 0)),
            patch.object(ds, "_load_data_ev", return_value=(mel, 0, 0)),
            patch.object(ds, "_preprocess", return_value=mel.unsqueeze(0)),
        ):
            return ds[0]

    def test_terminates_when_all_ood_texts_shorter_than_min_length(self):
        # Every pool entry is 2 chars; min_length=50 can never be met by a single pick.
        # The old while-loop-with-replacement would spin forever.
        pool = [("hi", [1, 2])] * 3
        ds = self._make_dataset(pool, min_length=50)
        with _time_limit(5):
            self._call_getitem(ds)

    def test_ood_text_is_accumulated_not_replaced(self):
        # "hello" → 5 chars, 5 indices (one per character, 'l' appears twice so same index).
        # min_length=20 requires at least 4 samples; samples are joined by a space (index
        # _SPACE_IDX) between each pair, so the result is more than 5 inner indices.
        pool = [("hello", [1, 2, 3, 3, 4])] * 20
        ds = self._make_dataset(pool, min_length=20)
        result = self._call_getitem(ds)
        ref_text_ood = result[4]
        inner = ref_text_ood[1:-1].tolist()  # strip leading/trailing boundary 0
        self.assertGreater(len(inner), 5)
        self.assertIn(_SPACE_IDX, inner)

    def test_timer_passes(self):
        with _time_limit(1):
            pass

    def test_timer_exceeded(self):
        with pytest.raises(TimeoutError):
            with _time_limit(1):
                time.sleep(2)


class TestOODIndicesFromPSV(unittest.TestCase):
    """OOD indices loaded from a real PSV file must map to valid symbol-table positions."""

    def setUp(self):
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2PretrainedConfig
        from styletts2.ev_config.text import EVStyleTTS2TextEncoder

        self.pretrained_symbols = StyleTTS2PretrainedConfig().pretrained_symbols
        encoder = EVStyleTTS2TextEncoder(TextConfig(), self.pretrained_symbols)

        ds = object.__new__(FilePathDataset)
        ds.preprocessed_dir = MagicMock()
        ds.OOD_min_length = 20
        ds.text_cleaner = TextCleaner()
        ds._ev_encoder = encoder
        ds._token_column = "character_tokens"
        ds.ood_texts = {}
        ds._load_ood_ev(ood_data_paths={"eng": _OOD_PSV}, ood_val_list=None)
        ds._ood_fallback_pool = [
            item for items in ds.ood_texts.values() for item in items
        ]
        ds.lang2id = None
        ds.data_list = [{"basename": "b0", "speaker": "spk0", "language": "eng"}]
        ds.df = pd.DataFrame([{"basename": "b0", "speaker": "spk0"}])
        self.ds = ds

    def _call_getitem(self):
        wave, text, mel = _dummy_tensors()
        with (
            patch.object(self.ds, "_load_tensor_ev", return_value=(wave, text, 0, 0)),
            patch.object(self.ds, "_load_data_ev", return_value=(mel, 0, 0)),
            patch.object(self.ds, "_preprocess", return_value=mel.unsqueeze(0)),
        ):
            return self.ds[0]

    def test_ood_indices_are_valid_symbol_table_positions(self):
        result = self._call_getitem()
        ref_text_ood = result[4]
        inner = ref_text_ood[1:-1].tolist()  # strip boundary tokens
        self.assertGreater(len(inner), 0, "OOD tensor has no inner indices")
        for idx in inner:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(self.pretrained_symbols))


class TestMultilingualLanguageId(unittest.TestCase):
    """language_id resolution in FilePathDataset._load_tensor_ev (EV mode)."""

    def _make_dataset(self, lang2id):
        ds = object.__new__(FilePathDataset)
        ds.lang2id = lang2id
        ds.speaker2id = None
        ds.sr = 24000
        ds.output_sampling_rate = 24000
        ds.silence_pad_samples = 0
        ds._ev_encoder = None
        ds._token_column = "character_tokens"
        ds.text_cleaner = TextCleaner()
        return ds

    def _load(self, ds, item):
        wave = np.zeros(1000)
        with (
            patch("styletts2.dataset.sf.read", return_value=(wave, ds.sr)),
            patch.object(ds, "_load_file", return_value="dummy.wav"),
        ):
            return ds._load_tensor_ev(item)

    def test_resolves_language_id_from_lang2id(self):
        ds = self._make_dataset({"eng": 0, "crk": 1})
        item = {
            "basename": "b0",
            "speaker": "spk0",
            "language": "crk",
            "characters": "hi",
        }
        _, _, _, language_id = self._load(ds, item)
        self.assertEqual(language_id, 1)

    def test_defaults_language_id_to_zero_when_monolingual(self):
        ds = self._make_dataset(None)
        item = {
            "basename": "b0",
            "speaker": "spk0",
            "language": "und",
            "characters": "hi",
        }
        _, _, _, language_id = self._load(ds, item)
        self.assertEqual(language_id, 0)


class TestCollaterLangs(unittest.TestCase):
    """The Collater must batch per-item language_ids into a `langs` tensor."""

    def test_langs_tensor_matches_per_item_language_ids(self):
        from styletts2.dataset import Collater

        def _item(language_id):
            mel = torch.zeros(80, 10)
            text = torch.zeros(3).long()
            ref_text = torch.zeros(3).long()
            ref_mel = torch.zeros(80, 10)
            wave = np.zeros(100)
            return (0, language_id, mel, text, ref_text, ref_mel, 0, "path", wave)

        batch = [_item(0), _item(1), _item(2)]
        collate = Collater(max_mel_length=10)
        result = collate(batch)
        langs = result[-1]
        self.assertEqual(sorted(langs.tolist()), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

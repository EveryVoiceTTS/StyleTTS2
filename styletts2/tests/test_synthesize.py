"""Tests for building synthesis entries from --text and --filelist input."""

import pytest


def test_build_text_entries_basic():
    from styletts2.cli.utils_heavy import build_text_entries

    entries = build_text_entries(
        ["Hello world", "How are you?"],
        "ref.wav",
        "LJ",
        "eng",
        5,
        1.0,
        0.3,
        0.7,
    )
    assert len(entries) == 2
    assert entries[0]["raw_text"] == "Hello world"
    assert entries[0]["basename"] == "Hello-world"
    assert entries[0]["speaker"] == "LJ"
    assert entries[0]["language"] == "eng"
    assert entries[0]["reference_path"] == "ref.wav"
    assert entries[1]["reference_path"] == "ref.wav"


def test_build_filelist_entries_row_values_override_defaults():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [
        {
            "basename": "LJ050-0269",
            "characters": "This is a test.",
            "speaker": "LJ",
            "language": "eng",
            "reference_path": "row_ref.wav",
        }
    ]
    entries = build_filelist_entries(
        rows, "default_ref.wav", "default_speaker", "und", 5, 1.0, 0.3, 0.7
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry["basename"] == "LJ050-0269"
    assert entry["raw_text"] == "This is a test."
    assert entry["speaker"] == "LJ"
    assert entry["language"] == "eng"
    assert entry["reference_path"] == "row_ref.wav"


def test_build_filelist_entries_missing_columns_fall_back_to_defaults():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [{"phones": "DH IH S"}]
    entries = build_filelist_entries(
        rows, "default_ref.wav", "default_speaker", "und", 5, 1.0, 0.3, 0.7
    )
    entry = entries[0]
    assert entry["raw_text"] == "DH IH S"
    assert entry["speaker"] == "default_speaker"
    assert entry["language"] == "und"
    assert entry["reference_path"] == "default_ref.wav"
    # No basename column, so one is auto-generated from the text
    assert entry["basename"] == "DH-IH-S"


def test_build_filelist_entries_reference_alias_column():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [{"characters": "hi", "reference": "aliased_ref.wav"}]
    entries = build_filelist_entries(
        rows, None, "default_speaker", "und", 5, 1.0, 0.3, 0.7
    )
    assert entries[0]["reference_path"] == "aliased_ref.wav"


def test_build_filelist_entries_missing_text_column_raises():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [{"speaker": "LJ"}]
    with pytest.raises(ValueError):
        build_filelist_entries(
            rows, "default_ref.wav", "default_speaker", "und", 5, 1.0, 0.3, 0.7
        )


def test_build_filelist_entries_missing_reference_raises():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [{"characters": "hi"}]
    with pytest.raises(ValueError):
        build_filelist_entries(rows, None, "default_speaker", "und", 5, 1.0, 0.3, 0.7)


_CHUNK_TEXT = "This is chunk one. This is chunk two. This is chunk three."
_CHUNK_SPLIT_PARAMS = (10, 25, "!?.", ":;,")


def test_build_text_entries_chunks_long_text():
    from styletts2.cli.utils_heavy import build_text_entries

    entries = build_text_entries(
        [_CHUNK_TEXT],
        "ref.wav",
        "LJ",
        "eng",
        5,
        1.0,
        0.3,
        0.7,
        split_text=True,
        split_params=_CHUNK_SPLIT_PARAMS,
    )
    assert len(entries) == 3
    assert [e["raw_text"] for e in entries] == [
        "This is chunk one.",
        "This is chunk two.",
        "This is chunk three.",
    ]
    # All chunks of the same input share one basename, so they can be
    # reassembled into a single output file.
    assert len({e["basename"] for e in entries}) == 1
    assert [e["is_last_input_chunk"] for e in entries] == [False, False, True]


def test_build_filelist_entries_chunks_long_text():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [{"basename": "row1", "characters": _CHUNK_TEXT}]
    entries = build_filelist_entries(
        rows,
        "default_ref.wav",
        "default_speaker",
        "und",
        5,
        1.0,
        0.3,
        0.7,
        split_text=True,
        split_params=_CHUNK_SPLIT_PARAMS,
    )
    assert len(entries) == 3
    assert all(e["basename"] == "row1" for e in entries)
    assert [e["is_last_input_chunk"] for e in entries] == [False, False, True]


def test_build_filelist_entries_no_chunking_by_default():
    from styletts2.cli.utils_heavy import build_filelist_entries

    rows = [{"characters": _CHUNK_TEXT}]
    entries = build_filelist_entries(
        rows, "default_ref.wav", "default_speaker", "und", 5, 1.0, 0.3, 0.7
    )
    assert len(entries) == 1
    assert entries[0]["raw_text"] == _CHUNK_TEXT
    assert entries[0]["is_last_input_chunk"] is True


class TestStyleTTS2PredictionWritingWavCallback:
    """Tests the accumulate-then-flush-on-last-chunk behaviour, mirroring
    FastSpeech2's chunk reassembly (fs2/tests/test_writing_callbacks.py)."""

    def test_flushes_only_on_last_chunk(self, tmp_path):
        import numpy as np

        from styletts2.cli.utils_heavy import StyleTTS2PredictionWritingWavCallback

        callback = StyleTTS2PredictionWritingWavCallback(tmp_path, global_step=0)

        chunk_outputs = [
            {
                "wav": np.zeros(100, dtype=np.float32),
                "sample_rate": 16000,
                "basename": "utt1",
                "speaker": "default",
                "language": "eng",
                "raw_text": "Chunk one. ",
                "is_last_input_chunk": False,
            },
            {
                "wav": np.zeros(150, dtype=np.float32),
                "sample_rate": 16000,
                "basename": "utt1",
                "speaker": "default",
                "language": "eng",
                "raw_text": "Chunk two.",
                "is_last_input_chunk": True,
            },
        ]

        for outputs in chunk_outputs:
            callback.on_predict_batch_end(None, None, outputs, None, 0)
            if not outputs["is_last_input_chunk"]:
                # No file written yet -- still accumulating.
                assert callback.last_file_written is None

        assert callback.last_file_written is not None
        written_files = list((tmp_path / "wav").glob("*.wav"))
        assert len(written_files) == 1

        import torchaudio

        waveform, _ = torchaudio.load(written_files[0])
        # Concatenated length of both chunks' waveforms.
        assert waveform.shape[-1] == 250

        # Accumulator state is reset after flushing.
        assert callback.full_wav.numel() == 0
        assert callback.full_text == ""
        assert callback.chunk_basenames == []

    def test_none_output_is_skipped(self, tmp_path):
        from styletts2.cli.utils_heavy import StyleTTS2PredictionWritingWavCallback

        callback = StyleTTS2PredictionWritingWavCallback(tmp_path, global_step=0)
        callback.on_predict_batch_end(None, None, None, None, 0)
        assert callback.last_file_written is None

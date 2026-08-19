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

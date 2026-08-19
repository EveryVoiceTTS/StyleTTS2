"""Tests for building synthesis entries from --text and --filelist input."""

import unittest


class TestBuildTextEntries(unittest.TestCase):
    def test_basic(self):
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
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["raw_text"], "Hello world")
        self.assertEqual(entries[0]["basename"], "Hello-world")
        self.assertEqual(entries[0]["speaker"], "LJ")
        self.assertEqual(entries[0]["language"], "eng")
        self.assertEqual(entries[0]["reference_path"], "ref.wav")
        self.assertEqual(entries[1]["reference_path"], "ref.wav")


class TestBuildFilelistEntries(unittest.TestCase):
    def test_row_values_override_defaults(self):
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
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["basename"], "LJ050-0269")
        self.assertEqual(entry["raw_text"], "This is a test.")
        self.assertEqual(entry["speaker"], "LJ")
        self.assertEqual(entry["language"], "eng")
        self.assertEqual(entry["reference_path"], "row_ref.wav")

    def test_missing_columns_fall_back_to_defaults(self):
        from styletts2.cli.utils_heavy import build_filelist_entries

        rows = [{"phones": "DH IH S"}]
        entries = build_filelist_entries(
            rows, "default_ref.wav", "default_speaker", "und", 5, 1.0, 0.3, 0.7
        )
        entry = entries[0]
        self.assertEqual(entry["raw_text"], "DH IH S")
        self.assertEqual(entry["speaker"], "default_speaker")
        self.assertEqual(entry["language"], "und")
        self.assertEqual(entry["reference_path"], "default_ref.wav")
        # No basename column, so one is auto-generated from the text
        self.assertEqual(entry["basename"], "DH-IH-S")

    def test_reference_alias_column(self):
        from styletts2.cli.utils_heavy import build_filelist_entries

        rows = [{"characters": "hi", "reference": "aliased_ref.wav"}]
        entries = build_filelist_entries(
            rows, None, "default_speaker", "und", 5, 1.0, 0.3, 0.7
        )
        self.assertEqual(entries[0]["reference_path"], "aliased_ref.wav")

    def test_missing_text_column_raises(self):
        from styletts2.cli.utils_heavy import build_filelist_entries

        rows = [{"speaker": "LJ"}]
        with self.assertRaises(ValueError):
            build_filelist_entries(
                rows, "default_ref.wav", "default_speaker", "und", 5, 1.0, 0.3, 0.7
            )

    def test_missing_reference_raises(self):
        from styletts2.cli.utils_heavy import build_filelist_entries

        rows = [{"characters": "hi"}]
        with self.assertRaises(ValueError):
            build_filelist_entries(
                rows, None, "default_speaker", "und", 5, 1.0, 0.3, 0.7
            )


if __name__ == "__main__":
    unittest.main()

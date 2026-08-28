"""Tests for the EveryVoice ↔ StyleTTS2 config/text integration."""

import pytest
from everyvoice.tests.stubs import capture_logs
from pydantic import ValidationError


class TestStyleTTS2PretrainedConfig:
    def test_default_pretrained_symbols_length(self):
        from styletts2.ev_config import StyleTTS2PretrainedConfig

        cfg = StyleTTS2PretrainedConfig()
        assert len(cfg.pretrained_symbols) == 178

    def test_default_pretrained_symbols_first_three(self):
        from styletts2.ev_config import StyleTTS2PretrainedConfig
        from styletts2.text_utils import symbols

        cfg = StyleTTS2PretrainedConfig()
        assert cfg.pretrained_symbols[:3] == list(symbols)[:3]


class TestStyleTTS2ModelConfig:
    def test_default_target_text_representation_level(self):
        from everyvoice.config.type_definitions import (
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.ev_config import StyleTTS2ModelConfig

        cfg = StyleTTS2ModelConfig()
        assert (
            cfg.target_text_representation_level
            == TargetTrainingTextRepresentationLevel.characters
        )

    def test_default_multilingual_fields(self):
        from styletts2.ev_config import StyleTTS2ModelConfig

        cfg = StyleTTS2ModelConfig()
        assert not cfg.multilingual
        assert cfg.language_embedding_dim == 64


_CONTACT = {"contact": {"contact_name": "Test", "contact_email": "test@test.com"}}


class TestToNativeConfig:
    def _make_config(self, **model_kwargs):
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2Config

        text = TextConfig()
        return StyleTTS2Config(text=text, **_CONTACT, **model_kwargs)

    def test_target_text_representation_in_data_params_characters(self):
        from styletts2.ev_config.translation import to_native_config

        cfg = self._make_config()
        native = to_native_config(cfg)
        assert native["data_params"]["target_text_representation"] == "characters"

    def test_target_text_representation_in_data_params_phones(self):
        from everyvoice.config.type_definitions import (
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.ev_config import StyleTTS2ModelConfig
        from styletts2.ev_config.translation import to_native_config

        cfg = self._make_config(
            model=StyleTTS2ModelConfig(
                target_text_representation_level=TargetTrainingTextRepresentationLevel.ipa_phones
            )
        )
        native = to_native_config(cfg)
        # TargetTrainingTextRepresentationLevel.ipa_phones.value == "phones"
        assert native["data_params"]["target_text_representation"] == "phones"

    def test_multilingual_fields_in_model_params(self):
        from styletts2.ev_config import StyleTTS2ModelConfig
        from styletts2.ev_config.translation import to_native_config

        cfg = self._make_config(
            model=StyleTTS2ModelConfig(multilingual=True, language_embedding_dim=32)
        )
        native = to_native_config(cfg)
        assert native["model_params"]["multilingual"]
        assert native["model_params"]["language_embedding_dim"] == 32


class TestEVStyleTTS2TextEncoder:
    @pytest.fixture(autouse=True)
    def _clear_warn_once(self):
        # _WARN_ONCE is module-level state; clear it so warning tests are independent.
        import styletts2.ev_config.text as _text_mod

        _text_mod._WARN_ONCE.clear()

    def _make_encoder(self):
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2PretrainedConfig
        from styletts2.ev_config.text import EVStyleTTS2TextEncoder

        symbols = StyleTTS2PretrainedConfig().pretrained_symbols
        return EVStyleTTS2TextEncoder(TextConfig(), symbols), symbols

    def test_encode_character_tokens(self):
        encoder, symbols = self._make_encoder()
        # "h/e/l/l/o" — these are all in the pretrained Latin+IPA set
        # Use simple latin chars that are definitely in the pretrained set
        indices = encoder.encode_token_sequence("h/e/l/l/o")
        assert len(indices) == 5
        # Each index should be a valid position in the symbol table
        for idx in indices:
            assert idx >= 0
            assert idx < len(symbols)

    def test_encode_ipa_phone_tokens(self):
        encoder, symbols = self._make_encoder()
        # IPA phones that are in StyleTTS2's _letters_ipa set
        indices = encoder.encode_token_sequence("h/ɛ/l/o/ʊ")
        assert len(indices) == 5
        for idx in indices:
            assert idx >= 0
            assert idx < len(symbols)

    def test_punctuation_remapping_tokens(self):
        from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

        encoder, symbols = self._make_encoder()
        tokens = {
            DEFAULT_PUNCTUATION_HASH["exclamations"]: "!",
            DEFAULT_PUNCTUATION_HASH["commas"]: ",",
        }
        for k, v in tokens.items():
            indices = encoder.encode_token_sequence(k)
            assert len(indices) == 1
            # "!" and "," should be in the pretrained symbol table
            assert symbols[indices[0]] == v

    def test_paren_dropped_with_warning(self):
        from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

        encoder, _ = self._make_encoder()
        paren_token = DEFAULT_PUNCTUATION_HASH["parentheses"]
        # <PAREN> has no StyleTTS2 equivalent — should be silently dropped
        with capture_logs() as warnings:
            indices = encoder.encode_token_sequence(paren_token)
        assert indices == []
        assert any("no mapping" in msg for msg in warnings)

    def test_unknown_token_dropped_with_warning(self):
        encoder, _ = self._make_encoder()
        with capture_logs() as warnings:
            indices = encoder.encode_token_sequence("<UNKNOWN_TOKEN_XYZ>")
        assert indices == []
        assert any("no mapping" in msg for msg in warnings)

    def test_empty_token_sequence(self):
        encoder, _ = self._make_encoder()
        indices = encoder.encode_token_sequence("")
        assert indices == []

    def test_mixed_valid_and_dropped_tokens(self):
        from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

        encoder, symbols = self._make_encoder()
        paren_token = DEFAULT_PUNCTUATION_HASH["parentheses"]
        # "h / <PAREN> / e" — PAREN should be dropped, h and e kept
        with capture_logs():
            indices = encoder.encode_token_sequence(f"h/{paren_token}/e")
        assert len(indices) == 2
        assert symbols[indices[0]] == "h"
        assert symbols[indices[1]] == "e"


class TestSymbolSubsetValidator:
    def _text_config_with_extra_symbols(self, *extra: str):
        """Return a TextConfig with *extra* symbols added as a custom symbol set."""
        from everyvoice.config.text_config import Symbols, TextConfig

        # Symbols uses extra="allow" — arbitrary keyword args become extra fields.
        symbols = Symbols(custom_letters=list(extra))
        return TextConfig(symbols=symbols)

    def test_validator_rejects_unknown_symbol(self):
        """A symbol not in the pretrained table should raise ValidationError,
        pointing the user at `everyvoice check pretrained-symbols` for suggested
        substitutions rather than computing them here (that requires
        numpy/scipy/panphon, which this validator must not import — it runs
        on every StyleTTS2Config instantiation, not just broken ones)."""
        from styletts2.ev_config import StyleTTS2Config

        # Korean character definitely not in StyleTTS2's Latin+IPA table
        bad_text = self._text_config_with_extra_symbols("가")
        with pytest.raises(ValidationError) as ctx:
            StyleTTS2Config(text=bad_text, **_CONTACT)
        message = str(ctx.value)
        assert "가" in message
        assert "everyvoice check pretrained-symbols" in message

    def test_validator_rejects_diphthong_not_in_pretrained(self):
        """A multi-character diphthong like 'oʊ' is not in the pretrained table,
        and the error still points to `everyvoice check pretrained-symbols`."""
        from styletts2.ev_config import StyleTTS2Config

        bad_text = self._text_config_with_extra_symbols("oʊ")
        with pytest.raises(ValidationError) as ctx:
            StyleTTS2Config(text=bad_text, **_CONTACT)
        message = str(ctx.value)
        assert "oʊ" in message
        assert "everyvoice check pretrained-symbols" in message

    def test_validator_does_not_import_utils_heavy(self):
        """The validator must stay cheap: it must not trigger the
        numpy/scipy/panphon import chain in everyvoice.text.utils_heavy, since
        it runs on every StyleTTS2Config instantiation, not just broken ones."""
        import sys

        from styletts2.ev_config import StyleTTS2Config

        sys.modules.pop("everyvoice.text.utils_heavy", None)
        bad_text = self._text_config_with_extra_symbols("가")
        with pytest.raises(ValidationError):
            StyleTTS2Config(text=bad_text, **_CONTACT)
        assert "everyvoice.text.utils_heavy" not in sys.modules

    def test_validator_accepts_valid_ipa_phones(self):
        """Single IPA phones in StyleTTS2's _letters_ipa set pass validation."""
        from styletts2.ev_config import StyleTTS2Config

        # "ɛ" and "ɝ" are both in StyleTTS2's pretrained _letters_ipa
        good_text = self._text_config_with_extra_symbols("ɛ", "ɝ")
        StyleTTS2Config(text=good_text, **_CONTACT)  # should not raise

    def test_validator_accepts_default_text_config(self):
        """The default TextConfig (no declared letters) passes validation trivially."""
        from everyvoice.config.text_config import TextConfig

        from styletts2.ev_config import StyleTTS2Config

        StyleTTS2Config(text=TextConfig(), **_CONTACT)


class TestEncodeTextForInference:
    """Tests for styletts2.utils.encode_text_for_inference, which replaced the
    old flat TextCleaner lookup at inference time with the same
    normalize/to_replace/G2P pipeline used at training time."""

    def _make_model(self, target_text_representation_level, **text_kwargs):
        import string

        from everyvoice.config.text_config import Symbols, TextConfig

        from styletts2.ev_config import StyleTTS2Config, StyleTTS2ModelConfig

        # Declare ascii letters and the specific single-character IPA phones
        # 'eng' g2p produces for "hello" (h ʌ l o ʊ) -- all present in
        # StyleTTS2's pretrained symbol table. Plain (non dataset-suffixed)
        # keys like these are included for every representation level.
        symbols = Symbols(ascii=list(string.ascii_lowercase), ipa=["ʌ", "ʊ"])
        config = StyleTTS2Config(
            text=TextConfig(symbols=symbols, **text_kwargs),
            model=StyleTTS2ModelConfig(
                target_text_representation_level=target_text_representation_level
            ),
            **_CONTACT,
        )

        class _FakeModel:
            pass

        model = _FakeModel()
        model.config = {"ev_config": config}
        return model

    def test_characters_trained_model_encodes_characters(self):
        from everyvoice.config.type_definitions import (
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.utils import encode_text_for_inference

        model = self._make_model(TargetTrainingTextRepresentationLevel.characters)
        tokens = encode_text_for_inference(model, "hello", "eng")
        assert tokens.numel() > 0

    def test_characters_trained_model_rejects_phones_input(self):
        from everyvoice.config.type_definitions import (
            DatasetTextRepresentation,
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.utils import encode_text_for_inference

        model = self._make_model(TargetTrainingTextRepresentationLevel.characters)
        with pytest.raises(ValueError):
            encode_text_for_inference(
                model, "h ɛ l oʊ", "eng", DatasetTextRepresentation.ipa_phones
            )

    def test_phones_trained_model_auto_phonemizes_characters(self):
        """A phones-trained model should still accept raw characters input by
        default, auto-phonemizing via g2p -- mirroring FastSpeech2's inference
        behaviour -- rather than requiring pre-phonemized text."""
        from everyvoice.config.type_definitions import (
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.utils import encode_text_for_inference

        model = self._make_model(TargetTrainingTextRepresentationLevel.ipa_phones)
        tokens = encode_text_for_inference(model, "hello", "eng")
        assert tokens.numel() > 0

    def test_phonological_features_not_implemented(self):
        from everyvoice.config.type_definitions import (
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.utils import encode_text_for_inference

        model = self._make_model(
            TargetTrainingTextRepresentationLevel.phonological_features
        )
        with pytest.raises(NotImplementedError):
            encode_text_for_inference(model, "hello", "eng")

    def test_missing_ev_config_raises(self):
        from styletts2.utils import encode_text_for_inference

        class _FakeModel:
            pass

        model = _FakeModel()
        model.config = {}
        with pytest.raises(ValueError):
            encode_text_for_inference(model, "hello", "eng")

    def test_to_replace_applied(self):
        """A to_replace rule should be applied before tokenization, so text
        containing the match encodes identically to text already containing
        the replacement."""
        from everyvoice.config.type_definitions import (
            TargetTrainingTextRepresentationLevel,
        )

        from styletts2.utils import encode_text_for_inference

        model_with_replace = self._make_model(
            TargetTrainingTextRepresentationLevel.characters,
            to_replace={"&": "and"},
        )
        model_plain = self._make_model(TargetTrainingTextRepresentationLevel.characters)
        with_replace = encode_text_for_inference(
            model_with_replace, "cats & dogs", "eng"
        )
        without_replace = encode_text_for_inference(model_plain, "cats and dogs", "eng")
        assert with_replace.tolist() == without_replace.tolist()


class TestCheckpointClassNameCompat:
    """check_and_upgrade_checkpoint accepts checkpoints written before the
    StyleTTS2Module -> StyleTTS2 rename and migrates their model_info."""

    def test_legacy_class_name_is_accepted_and_migrated(self):
        from styletts2.lightning import StyleTTS2

        model = StyleTTS2()  # config=None -> no heavy init
        checkpoint = {
            "model_info": {"name": "StyleTTS2Module", "version": "1.0"},
            "state_dict": {},
        }
        upgraded = model.check_and_upgrade_checkpoint(checkpoint)
        assert upgraded["model_info"]["name"] == "StyleTTS2"
        assert upgraded["model_info"]["version"] == "1.1"

    def test_wrong_class_name_still_rejected(self):
        from styletts2.lightning import StyleTTS2

        model = StyleTTS2()
        with pytest.raises(TypeError):
            model.check_and_upgrade_checkpoint(
                {"model_info": {"name": "FastSpeech2", "version": "1.0"}}
            )

    def test_alias_points_at_renamed_class(self):
        from styletts2.lightning import StyleTTS2, StyleTTS2Module

        assert StyleTTS2Module is StyleTTS2

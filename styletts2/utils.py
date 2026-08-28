import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from everyvoice import logger
from monotonic_align.core import maximum_path_c
from munch import Munch

MEL_MEAN = -4.0
MEL_STD = 4.0


def make_mel_transform(config):
    pp = config["preprocess_params"]
    sp = pp.get("spect_params", {})
    mp = pp.get("mel_params", {})
    return torchaudio.transforms.MelSpectrogram(
        n_mels=mp.get("n_mels", 80),
        n_fft=sp.get("n_fft", 2048),
        win_length=sp.get("win_length", 1200),
        hop_length=sp.get("hop_length", 300),
    )


def _load_reference_mel(path, target_sr, mel_transform):
    """Load and normalise a reference audio file into a mel spectrogram.

    Returns a tensor of shape ``[1, n_mels, T]`` on the same device as
    ``mel_transform``.
    """
    wave, sr = torchaudio.load(path)
    wave = wave.mean(0)
    if sr != target_sr:
        wave = torchaudio.functional.resample(wave, sr, target_sr)
    wave = wave.to(next(mel_transform.buffers()).device)
    mel = mel_transform(wave)
    return (torch.log(1e-5 + mel.unsqueeze(0)) - MEL_MEAN) / MEL_STD


def encode_text_for_inference(
    module,
    raw_text: str,
    language: "str | None",
    text_representation=None,
):
    """Normalize, optionally G2P, and tokenize raw text exactly like training-time
    preprocessing does, then translate the result into StyleTTS2 embedding indices.

    This mirrors what FastSpeech2's inference dataset does via
    ``Preprocessor.process_text`` (normalization + ``to_replace`` + G2P, driven by
    the model's own ``TextConfig``).

    Args:
        module: a ``StyleTTS2Module`` with an EveryVoice ``config["ev_config"]``.
        raw_text: the text to synthesize, as typed/passed by the caller.
        language: the language tag for the utterance (used to select a g2p engine
            and to look up ``to_replace``/cleaner rules); pass ``None`` for "und".
        text_representation: whether ``raw_text`` is already-phonemized IPA
            (``DatasetTextRepresentation.ipa_phones``) or raw ``characters``
            (the default). Only meaningful when the model wasn't trained on
            characters -- see the compatibility check below.

    Returns:
        A ``[1, T]`` LongTensor on CPU; move it to the target device before use.

    Raises:
        ValueError: if the checkpoint has no attached EveryVoice config, if
            ``text_representation`` is incompatible with the model's trained
            representation, or if the text produces no tokens.
        NotImplementedError: if the model was trained on phonological features,
            which StyleTTS2 inference does not support.
    """
    from everyvoice.config.type_definitions import (
        DatasetTextRepresentation,
        TargetTrainingTextRepresentationLevel,
    )
    from everyvoice.preprocessor.preprocessor import Preprocessor
    from everyvoice.text.text_processor import TextProcessor

    from .ev_config.text import EVStyleTTS2TextEncoder

    if text_representation is None:
        text_representation = DatasetTextRepresentation.characters

    if not isinstance(module.config, dict) or "ev_config" not in module.config:
        raise ValueError(
            "This checkpoint has no EveryVoice configuration attached "
            "('ev_config'), so text cannot be normalized/phonemized for "
            "inference. This feature requires a model trained via "
            "'everyvoice train text-to-wav'."
        )
    ev_config = module.config["ev_config"]
    target_level = ev_config.model.target_text_representation_level

    if target_level == TargetTrainingTextRepresentationLevel.phonological_features:
        raise NotImplementedError(
            "StyleTTS2 inference does not support phonological_features-trained "
            "checkpoints; only 'characters' and 'phones' are supported."
        )

    if (
        target_level == TargetTrainingTextRepresentationLevel.characters
        and text_representation != DatasetTextRepresentation.characters
    ):
        raise ValueError(
            f"Your model was trained on {target_level.value} but you provided "
            f"{text_representation.value} which is incompatible."
        )

    if not hasattr(module, "_text_processor"):
        module._text_processor = TextProcessor(
            ev_config.text, target_text_representation_level=target_level
        )
    if not hasattr(module, "_ev_encoder"):
        module._ev_encoder = EVStyleTTS2TextEncoder(
            ev_config.text, ev_config.pretrained.pretrained_symbols
        )

    if text_representation == DatasetTextRepresentation.arpabet:
        raise NotImplementedError(
            "StyleTTS2 inference does not yet support arpabet input; use "
            "'characters' or 'phones'."
        )

    if language is None:
        language = next(iter(getattr(module, "lang2id", None) or {}), None)
    item: dict = {"language": language or "und"}
    if text_representation == DatasetTextRepresentation.characters:
        item["characters"] = raw_text
    else:
        item["phones"] = raw_text

    characters, phones, _pfs = Preprocessor.process_text(
        item,
        text_processor=module._text_processor,
        use_pfs=False,
        encode_as_string=True,
    )
    logger.info(
        "StyleTTS2 inference text processing (pre-selection): "
        f"item={item!r} characters={characters!r} phones={phones!r}"
    )
    if target_level == TargetTrainingTextRepresentationLevel.ipa_phones:
        token_string = phones
    else:
        token_string = characters

    if not token_string:
        if (
            target_level == TargetTrainingTextRepresentationLevel.ipa_phones
            and text_representation == DatasetTextRepresentation.characters
        ):
            raise ValueError(
                f"Your model was trained on phones, but no g2p engine is "
                f"available for language '{item['language']}' to convert your "
                "characters input. Provide already-phonemized text with "
                "text_representation=phones instead."
            )
        raise ValueError(f"Text produced no tokens: {raw_text!r}")

    indices = module._ev_encoder.encode_token_sequence(token_string)
    if not indices:
        raise ValueError(f"Text produced no tokens: {raw_text!r}")

    logger.info(
        "StyleTTS2 inference text processing: "
        f"raw_text={raw_text!r} text_representation={text_representation.value!r} "
        f"target_text_representation_level={target_level.value!r} "
        f"token_string={token_string!r} indices={indices!r}"
    )

    return torch.LongTensor(indices).unsqueeze(0)


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "data/train_list.txt"
    if val_path is None:
        val_path = "data/val_list.txt"

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_list = f.readlines()
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_list = f.readlines()

    return train_list, val_list


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def log_print(message, logger):
    logger.info(message)
    print(message)

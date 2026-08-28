"""Core synthesis helpers for StyleTTS2.

Shared by the `everyvoice synthesize text-to-wav` CLI command
and the `everyvoice demo text-to-wav` Gradio app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from everyvoice import logger
from everyvoice.base_cli import command, default_typer_args
from everyvoice.base_cli.interfaces import typer_file_option
from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)


def load_styletts2_model(model_path: Path, device):
    """Load a StyleTTS2 Lightning module and mel transform from a checkpoint."""
    import torch

    from ..lightning import StyleTTS2Module
    from ..utils import make_mel_transform

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    module = StyleTTS2Module()
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    module.to(device)
    module.eval()

    mel_transform = make_mel_transform(module.config).to(device)
    return module, mel_transform


def load_reference_style(
    module,
    mel_transform,
    reference_path: Path,
    device,
):
    """Load a reference audio file and return a pre-computed style encoding.

    Runs ``_load_reference_mel`` then ``module._encode_reference``, returning
    ``ref_s`` of shape ``[1, 256]`` on ``device``.  Call this at startup to
    avoid re-computing on every synthesis request.
    """
    import torch

    from ..utils import (
        _load_reference_mel,
    )

    with torch.no_grad():
        ref_mel = _load_reference_mel(reference_path, module.sr, mel_transform).to(
            device
        )
        return module._encode_reference(ref_mel)


def synthesize_one(
    module,
    mel_transform,
    text: str,
    device,
    reference_path: Path,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    acoustic_blend: float = 0.3,
    prosody_blend: float = 0.7,
    language: str | None = None,
    text_representation: DatasetTextRepresentation | None = None,
):
    """Synthesize a single utterance and return a float32 numpy waveform.

    Works only with stage-2 (or finetune) checkpoints that include the
    diffusion sampler.  Stage-1 checkpoints will raise an AttributeError
    because ``module._sampler`` does not exist.

    ``language`` selects the language embedding for multilingual checkpoints
    (must be a key of ``module.lang2id``); ignored for monolingual checkpoints.
    ``text_representation`` indicates whether ``text`` is raw characters (the
    default) or already-phonemized IPA; see ``encode_text_for_inference``.
    """
    import torch

    from ..utils import (
        _load_reference_mel,
        encode_text_for_inference,
    )

    with torch.no_grad():
        tokens = encode_text_for_inference(
            module, text, language, text_representation
        ).to(device)

        input_lengths = torch.LongTensor([tokens.size(1)]).to(device)
        ref_mel = _load_reference_mel(reference_path, module.sr, mel_transform).to(
            device
        )

        lang_emb = None
        if hasattr(module, "language_embedding") and language in module.lang2id:
            lang_id = torch.LongTensor([module.lang2id[language]])
            lang_emb = module._lang_emb(lang_id)

        return module._synthesize_text(
            tokens,
            input_lengths,
            ref_mel=ref_mel,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            acoustic_blend=acoustic_blend,
            prosody_blend=prosody_blend,
            lang_emb=lang_emb,
        )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

app = typer.Typer(**default_typer_args)


@command(
    app,
    name="text-to-wav",
    short_help="Synthesize audio from text using a trained StyleTTS2 model",
)
def synthesize(
    model_path: Path = typer.Argument(
        ...,
        help="Path to a trained StyleTTS2 checkpoint (.ckpt).",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    reference: Path | None = typer_file_option(
        None,
        "--reference",
        "-r",
        help="Reference audio file used to extract speaker style. Required"
        " unless every row of --filelist provides its own 'reference' or"
        " 'reference_path' column.",
    ),
    text: list[str] = typer.Option(
        [],
        "--text",
        "-t",
        help="Text string(s) to synthesize. Repeat the flag for multiple utterances."
        " Use --filelist instead if you want to synthesize a lot of sentences or"
        " have different speaker/language/reference per sentence.",
    ),
    text_representation: DatasetTextRepresentation = typer.Option(
        DatasetTextRepresentation.characters,
        "--text-representation",
        help="The representation of the text passed via --text: 'characters' or"
        " 'phones' (already-phonemized IPA). The input type must be compatible"
        " with your model. Only applies to --text; --filelist rows determine"
        " this automatically from their 'characters'/'phones' column.",
    ),
    filelist: Path | None = typer_file_option(
        None,
        "--filelist",
        "-f",
        help="The path to a file containing a list of utterances (a.k.a filelist)."
        " Expected columns: 'basename', 'characters' (or 'phones'), 'speaker',"
        " 'language', and optionally 'reference' (or 'reference_path'). Any column"
        " that is absent falls back to the corresponding --speaker/--language/"
        "--reference CLI option. Use --text if you want to just synthesize one sample.",
    ),
    output_dir: Path = typer.Option(
        Path("synthesis_output"),
        "--output-dir",
        "-o",
        help="Directory where synthesized files will be written."
        " By default, filenames include the basename, speaker, language, and"
        " other metadata, e.g. 'LJ050-0269--LJ--eng--ckpt=100000--pred.wav'."
        " Use --simple-filenames to write just 'LJ050-0269.wav' instead.",
    ),
    output_type: list[SynthesizeOutputFormats] = typer.Option(
        [SynthesizeOutputFormats.wav],
        "--output-type",
        help="Output format(s) to produce.",
    ),
    simple_filenames: bool = typer.Option(
        False,
        "--simple-filenames",
        help="Write output filenames as just the basename and extension"
        " (e.g. 'LJ050-0269.wav') instead of the default."
        " Only use this if your basenames are unique across speakers and"
        " languages, otherwise outputs can overwrite each other.",
    ),
    accelerator: str = typer.Option(
        "auto",
        "--accelerator",
        help="Lightning accelerator: 'cpu', 'gpu', or 'auto'.",
    ),
    speaker: str = typer.Option(
        "default",
        "--speaker",
        "-s",
        help="Speaker label written into output filenames.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Language tag written into output filenames, and used to select "
        "a g2p engine/normalization rules for text processing. For "
        "multilingual checkpoints (must match a language seen during "
        "training), this also selects the language embedding. Defaults to "
        "the checkpoint's own (single) training language for monolingual "
        "checkpoints.",
    ),
    diffusion_steps: int = typer.Option(
        5,
        "--diffusion-steps",
        help="Number of diffusion sampling steps (higher = slower but smoother).",
    ),
    embedding_scale: float = typer.Option(
        1.0,
        "--embedding-scale",
        help="Classifier-free guidance scale for the diffusion sampler.",
    ),
    acoustic_blend: float = typer.Option(
        0.3,
        "--acoustic-blend",
        help="Blend weight for acoustic style (0 = pure reference, 1 = pure diffusion).",
    ),
    prosody_blend: float = typer.Option(
        0.7,
        "--prosody-blend",
        help="Blend weight for prosody style (0 = pure reference, 1 = pure diffusion).",
    ),
):
    """Synthesize audio from text using a trained StyleTTS2 model.

    Examples:

    **everyvoice synthesize text-to-wav logs_and_checkpoints/.../stage-2-last.ckpt \\
        --reference path/to/reference.wav \\
        --text "Hello world" --text "How are you?"**

    Or, for batch synthesis from a filelist:

    **everyvoice synthesize text-to-wav logs_and_checkpoints/.../stage-2-last.ckpt \\
        --reference path/to/reference.wav --filelist my_filelist.psv --simple-filenames**
    """
    # Do argument error checking before doing expensive imports
    if text and filelist:
        print(
            "Got arguments for both --text and --filelist."
            " You can only synthesize using one of these options",
            file=sys.stderr,
        )
        sys.exit(1)
    if not text and not filelist:
        print("You must define either --text or --filelist", file=sys.stderr)
        sys.exit(1)
    if text and reference is None:
        print(
            "Missing --reference option, which is required when using --text.",
            file=sys.stderr,
        )
        sys.exit(1)

    import lightning as L
    import torch

    from .utils_heavy import (
        StyleTTS2SynthesisDataModule,
        build_filelist_entries,
        build_text_entries,
        get_styletts2_synthesis_output_callbacks,
        get_styletts2_text_split_params,
    )

    device = torch.device(
        "cuda"
        if (
            accelerator == "gpu"
            or (accelerator == "auto" and torch.cuda.is_available())
        )
        else "cpu"
    )

    logger.info(f"Loading StyleTTS2 model from {model_path}")
    module, mel_transform = load_styletts2_model(model_path, device)
    module._mel_transform = mel_transform

    language = language or next(iter(module.lang2id.keys()), None)

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    global_step = int(state.get("global_step", 0))

    split_text, split_params = get_styletts2_text_split_params(
        module, language, text_representation
    )

    try:
        if text:
            entries = build_text_entries(
                text,
                str(reference),
                speaker,
                language,
                diffusion_steps,
                embedding_scale,
                acoustic_blend,
                prosody_blend,
                text_representation=text_representation,
                split_text=split_text,
                split_params=split_params,
            )
        else:
            assert filelist is not None
            filelist_loader = module.config["ev_config"].training.filelist_loader
            entries = build_filelist_entries(
                filelist_loader(filelist),
                str(reference) if reference else None,
                speaker,
                language,
                diffusion_steps,
                embedding_scale,
                acoustic_blend,
                prosody_blend,
                split_text=split_text,
                split_params=split_params,
            )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    callbacks = get_styletts2_synthesis_output_callbacks(
        output_type,
        output_dir,
        global_step,
        module.sr,
        simple_filenames=simple_filenames,
    )
    if not callbacks:
        logger.warning("No output format requested; nothing to do.")
        return

    datamodule = StyleTTS2SynthesisDataModule(entries)

    trainer = L.Trainer(
        accelerator=accelerator,
        callbacks=list(callbacks.values()),
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.predict(module, datamodule=datamodule)

    logger.info(f"Synthesis complete. Output saved to {output_dir}")

import multiprocessing as mp
from pathlib import Path

from . import core
from .ev_config import StyleTTS2Config


def load_config(
    config_file: Path | str,
) -> StyleTTS2Config:
    """Load a StyleTTS2 configuration from config_file.

    Your StyleTTS2 config file is called "config/everyvoice-text-to-wav.yaml" if it
    was created using the "everyvoice new-project" wizard.

    If you need to override any values, change them in the returned config object.

    Args:
        config_file (Path|str): StyleTTS2 configuration filename
    """
    return core.load_config(config_file=Path(config_file))


def preprocess(
    config: StyleTTS2Config,
    steps: list[str] = core.PREPROCESS_CATEGORIES,
    cpus: int = min(4, mp.cpu_count()),
    overwrite: bool = False,
    debug: bool = False,
) -> None:
    """Preprocess data for text-to-wav (StyleTTS2) training.

    The datasets to process are described in config.

    Args:
        config (StyleTTS2Config): your StyleTTS2 configuration
        steps (list[str]): steps to process, one or more of "audio", "text"
        cpus (int): how many CPUs to use for preprocessing
        overwrite (bool): if false, existing files will be kept and only new files will be generated;
                   if true, redo all preprocessing, even if files already exist
        debug (bool): enable debugging
    """
    core.preprocess(
        config=config,
        steps=steps,
        cpus=cpus,
        overwrite=overwrite,
        debug=debug,
    )

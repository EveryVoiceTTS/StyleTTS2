from everyvoice.config.shared_types import (
    BaseModelWithContact,
    BaseTrainingConfig,
    ConfigModel,
)

# DO_NOT_USE means that we can get this value from somewhere else in the EV config and do not need to add it to the StyleTTS2 config.

# DO_NOT_USE, log_dir: "Models/CRK_baseline_custom_asr_step_2"
# DO_NOT_USE, first_stage_path: "/home/aip000/u/StyleTTS2_mtessier/Models/CRK_baseline__custom_asr/epoch_1st_00198.pth"
# DO_NOT_USE, save_freq: 2
# DO_NOT_USE, log_interval: 10
# DO_NOT_USE, device: "cuda"
# DO_NOT_USE, epochs_1st: 200 # number of epochs for first stage training (pre-training)
# DO_NOT_USE, epochs_2nd: 100 # number of peochs for second stage training (joint training)
# DO_NOT_USE, batch_size: 4
# max_len: 350 # maximum number of frames
# DO_NOT_USE, pretrained_model: ""
# DO_NOT_USE, second_stage_load_pretrained: false # set to true if the pre-trained model is for 2nd stage
# DO_NOT_USE, load_only_params: false # set to true if do not want to load epoch numbers and optimizer parameters

# F0_path: "Utils/JDC/bst.t7"
# ASR_config: "Utils/ASR/config_crk_200.yml"
# ASR_path: "Utils/ASR/epoch_crk_00200.pth"
# PLBERT_dir: 'Utils/PLBERT/'

# DO_NOT_USE, data_params:
# DO_NOT_USE,   train_data: "Data/crk/train_list.txt"
# DO_NOT_USE,  val_data: "Data/crk/val_list.txt"
# DO_NOT_USE,  root_path: "Data/crk/"
#   OOD_data: "Data/crk/OOD_texts.txt"
#   min_length: 50 # sample until texts with this size are obtained for OOD texts

# DO_NOT_USE, preprocess_params:
# DO_NOT_USE,  sr: 24000
# DO_NOT_USE,  spect_params:
# DO_NOT_USE,    n_fft: 2048
# DO_NOT_USE,    win_length: 1200
# DO_NOT_USE,    hop_length: 300

# DO_NOT_USE, model_params:
# DO_NOT_USE,  multispeaker: true

#   dim_in: 64
#   hidden_dim: 512
#   max_conv_dim: 512
#   n_layer: 3
#   n_mels: 80

#   n_token: 178 # number of phoneme tokens
#   max_dur: 50 # maximum duration of a single phoneme
#   style_dim: 128 # style vector size

#   dropout: 0.2

#   # style diffusion model config
#   diffusion:
#     # transformer config
#     transformer:
#       head_features: 64
#       multiplier: 2

#     # diffusion distribution config
#     dist:
#       sigma_data: 0.2 # placeholder for estimate_sigma_data set to false
#       estimate_sigma_data: true # estimate sigma_data from the current batch if set to true
#       mean: -3.0
#       std: 1.0


class SLMAdvParams:
    min_len: int = Field(
        100, description="Minimum length (in samples) during SLM adversarial training"
    )
    max_len: int = Field(
        300, description="Max length (in samples) during SLM adversarial training"
    )
    batch_percentage: float = Field(
        0.5, description="Percentage of the batch to use in order to prevent OOM errors"
    )
    discriminator_iter: int = (
        Field(
            10,
            description="Update the discriminator after N iterations of the generator",
        ),
    )
    gradient_threshold: int = Field(
        5, description="Gradient norm threshold above which the gradient is scaled"
    )
    gradient_scaling_factor: float = Field(
        0.01,
        description="Gradient scaling factor for predictors from SLM discriminators. Applies above gradient_threshold.",
    )
    sigma: float = Field(1.5, description="sigma for differentiable duration modeling")


class SLMConfig(ConfigModel):
    model: str = Field(
        "microsoft/wavlm-base-plus",
        description="The HuggingFace location of a Speech Language Model",
    )
    sampling_rate: int = Field(16000, description="Sampling rate for SLM")
    hidden_dim: int = Field(768, description="Hidden size of SLM")
    layers: int = Field(13, description="Number of layers in SLM")
    input_dim: int = Field(
        64, description="Number of dimensions in initial SLM discriminator head"
    )


class TransformerConfig(ConfigModel):
    layers: int = Field(3, description="The number of layers in the Transformer.")
    heads: int = Field(8, description="The number of heads in the Transformer.")
    # head_features?
    # multiplier?


class DiffusionConfig(ConfigModel):
    embedding_mask_prob: float = Field(0.1, description="Embedding mask probability")
    transformer: TransformerConfig = Field(
        default_factory=TransformerConfig,
        description="The configuration for the diffusion transformer",
    )


class DecoderConfig(ConfigModel):
    resblock_kernel_sizes: list[int] = Field(
        [3, 7, 11],
        description="The kernel size of each convolutional layer in the resblock.",
    )
    upsample_rates: list[int] = Field(
        [10, 6],
        description="The stride of each convolutional layer in the upsampling module.",
    )
    upsample_initial_channel: int = Field(
        512,
        description="The number of dimensions to project the Mel inputs to before being passed to the resblock.",
    )
    resblock_dilation_sizes: list[list[int]] = Field(
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        description="The dilations of each convolution in each layer of the resblock.",
    )
    upsample_kernel_sizes: list[int] = Field(
        [20, 12],
        description="The kernel size of each convolutional layer in the upsampling module.",
    )
    gen_istft_n_fft: int = Field(20, description="TODO: see if necessary")
    gen_istft_hop_size: int = Field(5, description="TODO: see if necessary")
    istft_layer: bool = Field(
        True,
        description="Whether to predict phase and magnitude values and use an inverse Short-Time Fourier Transform instead of predicting a waveform directly. See Kaneko et. al. 2022: https://arxiv.org/abs/2203.02395",
    )


class StyleTTS2ModelConfig(ConfigModel):
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig,
        description="The configuration for the decoder module",
    )
    diffusion: DiffusionConfig = Field(
        default_factory=DiffusionConfig,
        description="The configuration for the diffusion module",
    )
    slm: SLMConfig = Field(
        default_factory=SLMConfig, description="The configuration for the SLM module"
    )


class StyleTTS2TrainingConfig(BaseTrainingConfig):
    bert_learning_rate: float = Field(
        1e-5, description="Learning rate for PLBERT text encoder"
    )
    finetune_learning_rate: float = Field(1e-5, description="TODO: confirm usage")
    lambda_mel: int = Field(5, description="Mel reconstruction loss")
    labmda_gen: int = Field(1, description="Generator loss")
    lambda_slm: int = (Field(1, description="SLM feature matching loss"),)
    lambda_mono: int = Field(1, description="Monotonic alignment loss (1st stage)")
    lambda_s2s: int = Field(1, description="Sequence-to-sequence loss (1st stage)")
    lambda_f0: int = Field(1, description="F0 reconstruction loss (2nd stage)")
    lambda_norm: int = Field(1, description="Norm reconstruction loss (2nd stage)")
    lambda_dur: int = Field(1, description="Duration reconstruction loss (2nd stage)")
    lambda_ce: int = Field(
        20, description="Duration predictor probability CE loss (2nd stage)"
    )
    lambda_sty: int = Field(1, description="Style reconstruction loss (2nd stage)")
    lambda_diff: int = Field(1, description="Score matching loss (2nd stage)")
    diff_epoch: int = Field(
        20, description="Starting epoch for style diffusion (2nd stage)"
    )
    joint_epoch: int = Field(
        50, description="Starting epoch for joing training (2nd stage)"
    )
    tma_epoch: int = Field(
        50, description="Starting epoch for monotonic aligner (1st stage)"
    )


class StyleTTS2Config(BaseModelWithContact):
    VERSION: Annotated[
        str,
        Field(init_var=False),
    ] = LATEST_VERSION

    model: StyleTTS2ModelConfig = Field(
        default_factory=StyleTTS2ModelConfig,
        description="The model configuration settings.",
    )
    path_to_model_config_file: Optional[FilePath] = Field(
        None, description="The path of a model configuration file."
    )

    training: StyleTTS2TrainingConfig = Field(
        default_factory=StyleTTS2TrainingConfig,
        description="The training configuration hyperparameters.",
    )
    path_to_training_config_file: Optional[FilePath] = Field(
        None, description="The path of a training configuration file."
    )

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="The preprocessing configuration, including information about audio settings.",
    )
    path_to_preprocessing_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    text: TextConfig = Field(
        default_factory=TextConfig, description="The text configuration."
    )
    path_to_text_config_file: Optional[FilePath] = Field(
        None, description="The path of a text configuration file."
    )

    @model_validator(mode="before")  # type: ignore
    def load_partials(self: Dict[Any, Any], info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing", "text"),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(path: Path) -> "StyleTTS2Config":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = StyleTTS2Config(**config)
        return config

    @model_validator(mode="before")
    @classmethod
    def check_and_upgrade_checkpoint(cls, data: Any) -> Any:
        """
        Check model's compatibility and possibly upgrade.
        """
        from packaging.version import Version

        ckpt_version = Version(data.get("VERSION", "0.0"))
        if ckpt_version > Version(LATEST_VERSION):
            raise ValueError(
                "Your config was created with a newer version of EveryVoice, please update your software."
            )
        # Successively convert model checkpoints to newer version.
        if ckpt_version < Version("1.0"):
            # Converting to 1.0 just requires setting the VERSION field
            data["VERSION"] = "1.0"

        return data

    # INPUT_TODO: initialize text with union of symbols from dataset

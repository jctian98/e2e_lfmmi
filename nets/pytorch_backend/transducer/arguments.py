"""Transducer model arguments."""

import ast
from distutils.util import strtobool


def add_encoder_general_arguments(group):
    """Define general arguments for encoder."""
    group.add_argument(
        "--etype",
        default="blstmp",
        type=str,
        choices=[
            "custom",
            "lstm",
            "blstm",
            "lstmp",
            "blstmp",
            "vgglstmp",
            "vggblstmp",
            "vgglstm",
            "vggblstm",
            "gru",
            "bgru",
            "grup",
            "bgrup",
            "vgggrup",
            "vggbgrup",
            "vgggru",
            "vggbgru",
        ],
        help="Type of encoder network architecture",
    )
    group.add_argument(
        "--dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for the encoder",
    )

    return group


def add_rnn_encoder_arguments(group):
    """Define arguments for RNN encoder."""
    group.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers (for shared recognition part "
        "in multi-speaker asr mode)",
    )
    group.add_argument(
        "--eunits",
        "-u",
        default=300,
        type=int,
        help="Number of encoder hidden units",
    )
    group.add_argument(
        "--eprojs", default=320, type=int, help="Number of encoder projection units"
    )
    group.add_argument(
        "--subsample",
        default="1",
        type=str,
        help="Subsample input frames x_y_z means subsample every x frame "
        "at 1st layer, every y frame at 2nd layer etc.",
    )

    return group


def add_custom_encoder_arguments(group):
    """Define arguments for Custom encoder."""
    group.add_argument(
        "--enc-block-arch",
        type=eval,
        action="append",
        default=None,
        help="Encoder architecture definition by blocks",
    )
    group.add_argument(
        "--enc-block-repeat",
        default=0,
        type=int,
        help="Repeat N times the provided encoder blocks if N > 1",
    )
    group.add_argument(
        "--custom-enc-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "vgg2l", "linear", "embed", "null"],
        help="Custom encoder input layer type",
    )
    group.add_argument(
        "--custom-enc-positional-encoding-type",
        type=str,
        default="abs_pos",
        choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
        help="Custom encoder positional encoding layer type",
    )
    group.add_argument(
        "--custom-enc-self-attn-type",
        type=str,
        default="self_attn",
        choices=["self_attn", "rel_self_attn"],
        help="Custom encoder self-attention type",
    )
    group.add_argument(
        "--custom-enc-pw-activation-type",
        type=str,
        default="relu",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Custom encoder pointwise activation type",
    )
    group.add_argument(
        "--custom-enc-conv-mod-activation-type",
        type=str,
        default="swish",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Custom encoder convolutional module activation type",
    )

    return group


def add_decoder_general_arguments(group):
    """Define general arguments for encoder."""
    group.add_argument(
        "--dtype",
        default="lstm",
        type=str,
        choices=["lstm", "gru", "custom"],
        help="Type of decoder to use",
    )
    group.add_argument(
        "--dropout-rate-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder",
    )
    group.add_argument(
        "--dropout-rate-embed-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder embedding layer",
    )

    return group


def add_rnn_decoder_arguments(group):
    """Define arguments for RNN decoder."""
    group.add_argument(
        "--dec-embed-dim",
        default=320,
        type=int,
        help="Number of decoder embeddings dimensions",
    )
    group.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )

    return group


def add_custom_decoder_arguments(group):
    """Define arguments for Custom decoder."""
    group.add_argument(
        "--dec-block-arch",
        type=eval,
        action="append",
        default=None,
        help="Custom decoder blocks definition",
    )
    group.add_argument(
        "--dec-block-repeat",
        default=1,
        type=int,
        help="Repeat N times the provided decoder blocks if N > 1",
    )
    group.add_argument(
        "--custom-dec-input-layer",
        type=str,
        default="embed",
        choices=["linear", "embed"],
        help="Custom decoder input layer type",
    )
    group.add_argument(
        "--custom-dec-pw-activation-type",
        type=str,
        default="relu",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Custom decoder pointwise activation type",
    )

    return group


def add_custom_training_arguments(group):
    """Define arguments for training with Custom architecture."""
    group.add_argument(
        "--transformer-warmup-steps",
        default=25000,
        type=int,
        help="Optimizer warmup steps",
    )
    group.add_argument(
        "--transformer-lr",
        default=10.0,
        type=float,
        help="Initial value of learning rate",
    )

    return group


def add_transducer_arguments(group):
    """Define general arguments for transducer model."""
    group.add_argument(
        "--trans-type",
        default="warp-transducer",
        type=str,
        choices=["warp-transducer", "warp-rnnt"],
        help="Type of transducer implementation to calculate loss.",
    )
    group.add_argument(
        "--transducer-weight",
        default=1.0,
        type=float,
        help="Weight of transducer loss when auxiliary task is used.",
    )
    group.add_argument(
        "--joint-dim",
        default=320,
        type=int,
        help="Number of dimensions in joint space",
    )
    group.add_argument(
        "--joint-activation-type",
        type=str,
        default="tanh",
        choices=["relu", "tanh", "swish"],
        help="Joint network activation type",
    )
    group.add_argument(
        "--score-norm",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize transducer scores by length",
    )

    return group


def add_auxiliary_task_arguments(group):
    """Add arguments for auxiliary task."""
    group.add_argument(
        "--aux-task-type",
        nargs="?",
        default=None,
        choices=["default", "symm_kl_div", "both"],
        help="Type of auxiliary task.",
    )
    group.add_argument(
        "--aux-task-layer-list",
        default=None,
        type=ast.literal_eval,
        help="List of layers to use for auxiliary task.",
    )
    group.add_argument(
        "--aux-task-weight",
        default=0.3,
        type=float,
        help="Weight of auxiliary task loss.",
    )
    group.add_argument(
        "--aux-ctc",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to use CTC as auxiliary task.",
    )
    group.add_argument(
        "--aux-ctc-weight",
        default=1.0,
        type=float,
        help="Weight of auxiliary task loss",
    )
    group.add_argument(
        "--aux-ctc-dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for auxiliary CTC",
    )
    group.add_argument(
        "--aux-cross-entropy",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to use CE as auxiliary task for the prediction network.",
    )
    group.add_argument(
        "--aux-cross-entropy-smoothing",
        default=0.0,
        type=float,
        help="Smoothing rate for cross-entropy. If > 0, enables label smoothing loss.",
    )
    group.add_argument(
        "--aux-cross-entropy-weight",
        default=0.5,
        type=float,
        help="Weight of auxiliary task loss",
    )
    group.add_argument(
        "--aux-mmi",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to use mmi as auxiliary task.",
    )
    group.add_argument(
        "--aux-mmi-weight",
        default=0.5,
        type=float,
        help="Weight of auxiliary mmi loss",
    )
    group.add_argument(
        "--aux-mmi-dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for auxiliary mmi",
    )
    group.add_argument(
        "--aux-mmi-type",
        type=str,
        choices=['mmi', 'phonectc'],
        default='mmi',
        help="LF-MMI or CTC",
    )
    group.add_argument(
        "--aux-mbr",
        type=strtobool,
        nargs="?",
        default=False,
        help="Whether to use mbr as auxiliary task.",
    )
    group.add_argument(
        "--aux-mbr-weight",
        default=1.0,
        type=float,
        help="Weight of auxiliary mbr loss",
    )
    group.add_argument(
        "--aux-mbr-beam",
        default=2,
        type=int,
        help="Number of hypothesis for MBR loss computation",
    )

    return group

def add_att_scorer_arguments(group):
    """
    Argument mainly copied from: espnet.nets.pytorch_backend.transformer.argument
    We only copy the argument for attention decoder / rescorer
    All arguments are added with prefix 'att', which means RNN-T attention scorer only
    """
    group.add_argument(
        "--att-scorer-weight",
        default=0.0,
        type=float,
        help="weight of attention scorer loss",
    )
    group.add_argument(
        "--att-decoder-selfattn-layer-type",
        type=str,
        default="selfattn",
        choices=[
            "selfattn",
            "lightconv",
            "lightconv2d",
            "dynamicconv",
            "dynamicconv2d",
            "light-dynamicconv2d",
        ],
        help="transformer decoder self-attention layer type",
    )
    group.add_argument(
        "--att-adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    group.add_argument(
        "--att-aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    group.add_argument(
        "--att-wshare",
        default=4,
        type=int,
        help="Number of parameter shargin for lightweight convolution",
    )
    group.add_argument(
        "--att-ldconv-decoder-kernel-length",
        default="11_13_15_17_19_21",
        type=str,
        help="kernel size for lightweight/dynamic convolution: "
        'Decoder side. For example, "21_23_25" means kernel length 21 for '
        "First layer, 23 for Second layer and so on.",
    )
    group.add_argument(
        "--att-ldconv-usebias",
        type=strtobool,
        default=False,
        help="use bias term in lightweight/dynamic convolution",
    )
    group.add_argument(
        "--att-dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--att-dunits", default=320, type=int, help="Number of decoder hidden units"
    )
    group.add_argument(
        "--att-attn-dropout-rate",
        default=None,
        type=float,
        help="dropout in transformer attention. use --dropout-rate if None is set",
    )
    group.add_argument(
        "--att-dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for the encoder",
    )
    group.add_argument(
        "--att-length-normalized-loss",
        default=True,
        type=strtobool,
        help="normalize loss by length",
    )
    return group


def add_transducer_code_switch_arguments(group):
    """Define general arguments for transducer model."""
    group.add_argument(
        "--cs-is-pretrain",
        default=False,
        type=strtobool,
        help="If true, ignore decoder loss",
    )
    group.add_argument(
        "--cs-share-encoder",
        default=False,
        type=strtobool,
        help="If true, use a shared encoder before the language-specific encoder",
    )
    group.add_argument(
        "--cs-share-encoder-layers",
        default=9,
        type=int,
        help="If true, number of layers in shared encoder",
    )
    group.add_argument(
        "--cs-chn-start",
        default=5,
        type=int,
        help="start index of chn symbols in dict",
    )
    group.add_argument(
        "--cs-eng-start",
        default=4302,
        type=int,
        help="start index of eng symbols in dict",
    )
    group.add_argument(
        "--cs-use-adversial-examples",
        default=False,
        type=strtobool,
        help="If true, mask symbols not from this language",
    )
    group.add_argument(
        "--cs-is-ctc-decoder",
        default=False,
        type=strtobool,
        help="If true, the fine tuning system is on CTC rather than RNNT",
    )
    group.add_argument(
        "--cs-use-mask-predictor",
        default=False,
        type=strtobool,
        help="If true, use a mask-filter process in combine function",
    )
    group.add_argument(
        "--cs-lang-weight",
        default=0.0,
        type=float,
        help="weight of language classificiation",
    )
    group.add_argument(
        "--cs-decoder-expert",
        default=False,
        type=strtobool,
        help="If true, use decoder expert",
    )
    return group

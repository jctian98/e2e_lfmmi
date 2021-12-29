from typing import Optional

from torch import nn
from torch.utils.tensorboard import SummaryWriter


class AcousticModel(nn.Module):
    """
    AcousticModel specifies the common attributes/methods that
    will be exposed by all Snowfall acoustic model networks.
    Think of it as of an interface class.
    """

    # A.k.a. the input feature dimension.
    num_features: int

    # A.k.a. the output dimension (could be the number of phones or
    # characters in the vocabulary).
    num_classes: int

    # When greater than one, the networks output sequence length will be
    # this many times smaller than the input sequence length.
    subsampling_factor: int

    def write_tensorboard_diagnostics(
            self,
            tb_writer: SummaryWriter,
            global_step: Optional[int] = None
    ):
        """
        Collect interesting diagnostic info about the model and write to to TensorBoard.
        Unless overridden, logs nothing.

        :param tb_writer: a TensorBoard ``SummaryWriter`` instance.
        :param global_step: optional number of total training steps done so far.
        """
        pass

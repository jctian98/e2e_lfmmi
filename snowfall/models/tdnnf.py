#!/usr/bin/env python3

# Copyright 2021 John's Hopkins University (author: Piotr Żelasko)
# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from snowfall.models import AcousticModel
from snowfall.training.diagnostics import measure_semiorthogonality, measure_weight_norms

"""
CAUTION! This model is not fully ported from Kaldi. It will converge, but its training
is still unstable and it seems to underperform its Kaldi counterpart.
We expect to improve this going forward.
"""


def tdnnf_optimizer(
        model: nn.Module,
        learning_rate: float = 5e-5,
        momentum: float = 0.9,
        weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """
    This is an example of an optimizer with parameter/layer-specific learning rates.
    We don't use it by default but it can be helpful in tuning the training of a specific model.
    """
    out_layer_keys = {'output_affine.weight', 'output_affine.bias', 'prefinal_l.weight', 'prefinal_l.bias'}
    return torch.optim.SGD([
        # Default optimization settings
        {'params': [p for key, p in model.named_parameters() if key not in out_layer_keys]},
        # Output layer may need smaller LR
        {'params': [model.output_affine.weight], 'lr': learning_rate * 0.5},
        {'params': [model.output_affine.bias], 'lr': learning_rate * 0.1},
    ],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )


class Tdnnf1a(AcousticModel):
    """
    This is a PyTorch implementation of a standard Kaldi TDNN-F model architecture.
    The default configuration is based on the Kaldi nnet3 xconfig below,
    except it doesn't use an LDA transform.
    Note that unlike Kaldi models it does not have a cross-entropy output layer,
    as Snowfall does not support alignments in training at this time.

    .. code-block:

        input dim=43 name=input
        fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=exp/chain_cleaned_1c/tdnn1c_sp/configs/lda.mat
        relu-batchnorm-dropout-layer name=tdnn1 l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true dim=1024
        tdnnf-layer name=tdnnf2 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
        tdnnf-layer name=tdnnf3 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
        tdnnf-layer name=tdnnf4 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
        tdnnf-layer name=tdnnf5 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=0
        tdnnf-layer name=tdnnf6 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf7 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf8 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf9 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf10 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf11 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf12 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        tdnnf-layer name=tdnnf13 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
        linear-component name=prefinal-l dim=256 l2-regularize=0.008 orthonormal-constraint=-1.0
        prefinal-layer name=prefinal-chain input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
        output-layer name=output include-log-softmax=false dim=3456 l2-regularize=0.002
    """

    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_dim=1024,
                 bottleneck_dim=128,
                 prefinal_bottleneck_dim=256,
                 kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
                 subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
                 subsampling_factor=3):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor

        # at present, we support only frame_subsampling_factor to be 3
        assert self.subsampling_factor == 3

        assert len(kernel_size_list) == len(subsampling_factor_list)
        num_layers = len(kernel_size_list)

        self.ortho_constrain_count = 0

        self.input_batch_norm = nn.BatchNorm1d(num_features=self.num_features, affine=False)

        self.tdnn1 = TDNN(input_dim=self.num_features, hidden_dim=hidden_dim)

        tdnnfs = []
        for i in range(num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = FactorizedTDNN(dim=hidden_dim,
                                   bottleneck_dim=bottleneck_dim,
                                   kernel_size=kernel_size,
                                   subsampling_factor=subsampling_factor,
                                   cnn_padding=int(subsampling_factor == 1))
            tdnnfs.append(layer)

        # tdnnfs requires [N, C, T]
        self.tdnnfs = nn.ModuleList(tdnnfs)

        # prefinal_l affine requires [N, C, T]
        self.prefinal_l = OrthonormalLinear(
            dim=hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            kernel_size=1)

        # prefinal_chain requires [N, C, T]
        self.prefinal_chain = PrefinalLayer(big_dim=hidden_dim,
                                            small_dim=prefinal_bottleneck_dim)

        # output_affine requires [N, T, C]
        self.output_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                       out_features=self.num_classes)

        self.register_forward_pre_hook(constrain_orthonormal_hook)

    def forward(self, x, dropout=0.):
        # input x is of shape: [batch_size, feat_dim, seq_len] = [N, C, T]
        assert x.ndim == 3

        # at this point, x is [N, C, T]
        x = self.input_batch_norm(x)

        # at this point, x is [N, C, T]
        x = self.tdnn1(x, dropout=dropout)

        # tdnnf requires input of shape [N, C, T]
        for layer in self.tdnnfs:
            x = layer(x, dropout=dropout)

        # at this point, x is [N, C, T]
        x = self.prefinal_l(x)

        # at this point, x is [N, C, T]
        nnet_output = self.prefinal_chain(x)
        # at this point, nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2, 1)
        # at this point, nnet_output is [N, T, C]
        nnet_output = self.output_affine(nnet_output)
        nnet_output = F.log_softmax(nnet_output, dim=2)
        # we return nnet_output [N, C, T]
        nnet_output = nnet_output.permute(0, 2, 1)
        return nnet_output

    def write_tensorboard_diagnostics(self, tb_writer: SummaryWriter, global_step: Optional[int] = None):
        tb_writer.add_scalars(
            'train/semiorthogonality_score',
            measure_semiorthogonality(self),
            global_step=global_step
        )
        tb_writer.add_scalars(
            'train/weight_l2_norms',
            measure_weight_norms(self, norm='l2'),
            global_step=global_step
        )
        tb_writer.add_scalars(
            'train/weight_max_norms',
            measure_weight_norms(self, norm='linf'),
            global_step=global_step
        )


def constrain_orthonormal_hook(model, unused_x):
    if not model.training:
        return

    model.ortho_constrain_count = (model.ortho_constrain_count + 1) % 2
    if model.ortho_constrain_count != 0:
        return

    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, 'constrain_orthonormal'):
                m.constrain_orthonormal()


def _constrain_orthonormal_internal(M):
    '''
    Refer to
        void ConstrainOrthonormalInternal(BaseFloat scale, CuMatrixBase<BaseFloat> *M)
    from
        https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc#L982
    Note that we always use the **floating** case.
    '''
    assert M.ndim == 2

    num_rows = M.size(0)
    num_cols = M.size(1)

    assert num_rows <= num_cols

    # P = M * M^T
    P = torch.mm(M, M.t())
    P_PT = torch.mm(P, P.t())

    trace_P = torch.trace(P)
    trace_P_P = torch.trace(P_PT)

    scale = torch.sqrt(trace_P_P / trace_P)

    ratio = trace_P_P * num_rows / (trace_P * trace_P)
    assert ratio > 0.99

    update_speed = 0.125

    if ratio > 1.02:
        update_speed *= 0.5
        if ratio > 1.1:
            update_speed *= 0.5

    identity = torch.eye(num_rows, dtype=P.dtype, device=P.device)
    P = P - scale * scale * identity

    alpha = update_speed / (scale * scale)
    M = M - 4 * alpha * torch.mm(P, M)
    return M


class SharedDimScaleDropout(nn.Module):
    def __init__(self, dim=1):
        '''
        Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
        '''
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', torch.tensor(0.))

    def forward(self, x, alpha=0.0):
        if self.training and alpha > 0.:
            # sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
            tied_mask_shape = list(x.shape)
            tied_mask_shape[self.dim] = 1
            repeats = [1 if i != self.dim else x.shape[self.dim]
                       for i in range(len(x.shape))]
            return x * self.mask.repeat(tied_mask_shape).uniform_(1 - 2 * alpha, 1 + 2 * alpha).repeat(repeats)
            # expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
        return x


class OrthonormalLinear(nn.Module):

    def __init__(self, dim, bottleneck_dim, kernel_size):
        super().__init__()
        # WARNING(fangjun): kaldi uses [-1, 0] for the first linear layer
        # and [0, 1] for the second affine layer;
        # we use [-1, 0, 1] for the first linear layer if time_stride == 1

        self.kernel_size = kernel_size

        # conv requires [N, C, T]
        self.conv = nn.Conv1d(in_channels=dim,
                              out_channels=bottleneck_dim,
                              kernel_size=kernel_size,
                              bias=False)

    def forward(self, x):
        # input x is of shape: [batch_size, feat_dim, seq_len] = [N, C, T]
        assert x.ndim == 3
        x = self.conv(x)
        return x

    def constrain_orthonormal(self):
        state_dict = self.conv.state_dict()
        w = state_dict['weight']
        # w is of shape [out_channels, in_channels, kernel_size]
        out_channels = w.size(0)
        in_channels = w.size(1)
        kernel_size = w.size(2)

        w = w.reshape(out_channels, -1)

        num_rows = w.size(0)
        num_cols = w.size(1)

        need_transpose = False
        if num_rows > num_cols:
            w = w.t()
            need_transpose = True

        w = _constrain_orthonormal_internal(w)

        if need_transpose:
            w = w.t()

        w = w.reshape(out_channels, in_channels, kernel_size)

        state_dict['weight'] = w
        self.conv.load_state_dict(state_dict)


class PrefinalLayer(nn.Module):

    def __init__(self, big_dim, small_dim):
        super().__init__()
        self.affine = nn.Linear(in_features=small_dim, out_features=big_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=big_dim, affine=False)
        self.linear = OrthonormalLinear(dim=big_dim,
                                        bottleneck_dim=small_dim,
                                        kernel_size=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=small_dim, affine=False)

    def forward(self, x):
        # x is [N, C, T]
        x = x.permute(0, 2, 1)

        # at this point, x is [N, T, C]

        x = self.affine(x)
        x = F.relu(x)

        # at this point, x is [N, T, C]

        x = x.permute(0, 2, 1)

        # at this point, x is [N, C, T]

        x = self.batchnorm1(x)

        x = self.linear(x)

        x = self.batchnorm2(x)

        return x


class TDNN(nn.Module):
    '''
    This class implements the following topology in kaldi:
      relu-batchnorm-dropout-layer name=tdnn1 dropout-per-dim-continuous=true dim=1024
    '''

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # affine conv1d requires [N, C, T]
        self.affine = nn.Conv1d(in_channels=input_dim,
                                out_channels=hidden_dim,
                                kernel_size=1)

        # tdnn1_batchnorm requires [N, C, T]
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_dim,
                                        affine=False)

        self.dropout = SharedDimScaleDropout(dim=2)

    def forward(self, x, dropout=0.):
        # input x is of shape: [batch_size, feat_dim, seq_len] = [N, C, T]
        x = self.affine(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x, alpha=dropout)
        # return shape is [N, C, T]
        return x


class FactorizedTDNN(nn.Module):
    '''
    This class implements the following topology in kaldi:
      tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
    References:
        - http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        - ConstrainOrthonormalInternal() from
          https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-utils.cc#L982
    '''

    def __init__(self,
                 dim,
                 bottleneck_dim,
                 kernel_size,
                 subsampling_factor,
                 bypass_scale=0.66,
                 cnn_padding=1):
        super().__init__()

        assert abs(bypass_scale) <= 1

        self.bypass_scale = bypass_scale

        self.s = subsampling_factor

        # linear requires [N, C, T]
        self.linear = OrthonormalLinear(dim=dim,
                                        bottleneck_dim=bottleneck_dim,
                                        kernel_size=kernel_size)

        # affine requires [N, C, T]
        # WARNING(fangjun): we do not use nn.Linear here
        # since we want to use `stride`
        self.affine = nn.Conv1d(in_channels=bottleneck_dim,
                                out_channels=dim,
                                kernel_size=1,
                                stride=subsampling_factor,
                                padding=cnn_padding)

        # batchnorm requires [N, C, T]
        self.batchnorm = nn.BatchNorm1d(num_features=dim, affine=False)

        self.dropout = SharedDimScaleDropout(dim=2)

    def forward(self, x, dropout=0.):
        # input x is of shape: [batch_size, feat_dim, seq_len] = [N, C, T]
        assert x.ndim == 3

        # save it for skip connection
        input_x = x

        x = self.linear(x)
        # at this point, x is [N, C, T]

        x = self.affine(x)
        # at this point, x is [N, C, T]

        x = F.relu(x)

        # at this point, x is [N, C, T]

        x = self.batchnorm(x)

        # at this point, x is [N, C, T]

        x = self.dropout(x, alpha=dropout)

        if self.linear.kernel_size > 1:
            # padding takes care of keeping the shapes correct
            x = self.bypass_scale * input_x + x
        else:
            x = self.bypass_scale * input_x[:, :, ::self.s] + x
        return x



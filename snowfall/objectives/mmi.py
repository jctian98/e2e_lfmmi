from typing import List, Tuple

import torch
from torch import nn

import k2

from snowfall.objectives.common import get_tot_objf_and_num_frames
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler


def _compute_mmi_loss_exact_optimized(
        nnet_output: torch.Tensor,
        texts: List[str],
        supervision_segments: torch.Tensor,
        graph_compiler: MmiTrainingGraphCompiler,
        P: k2.Fsa,
        den_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    The function name contains `exact`, which means it uses a version of
    intersection without pruning.

    `optimized` in the function name means this function is optimized
    in that it calls k2.intersect_dense only once

    Note:
      It is faster at the cost of using more memory.

    Args:
      nnet_output:
        A 3-D tensor of shape [N, T, C]
      texts:
        The transcript. Each element consists of space(s) separated words.
      supervision_segments:
        A 2-D tensor that will be passed to :func:`k2.DenseFsaVec`.
      graph_compiler:
        Used to build num_graphs and den_graphs
      P:
        Represents a bigram Fsa.
      den_scale:
        The scale applied to the denominator tot_scores.
    '''

    num_graphs, den_graphs = graph_compiler.compile(texts,
                                                    P,
                                                    replicate_den=False)

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    device = num_graphs.device

    num_fsas = num_graphs.shape[0]
    assert dense_fsa_vec.dim0() == num_fsas

    assert den_graphs.shape[0] == 1

    # the aux_labels of num_graphs is k2.RaggedInt
    # but it is torch.Tensor for den_graphs.
    #
    # The following converts den_graphs.aux_labels
    # from torch.Tensor to k2.RaggedInt so that
    # we can use k2.append() later
    den_graphs.convert_attr_to_ragged_(name='aux_labels')

    # The motivation to concatenate num_graphs and den_graphs
    # is to reduce the number of calls to k2.intersect_dense.
    num_den_graphs = k2.cat([num_graphs, den_graphs])

    # NOTE: The a_to_b_map in k2.intersect_dense must be sorted
    # so the following reorders num_den_graphs.
    #
    # The following code computes a_to_b_map

    # [0, 1, 2, ... ]
    num_graphs_indexes = torch.arange(num_fsas, dtype=torch.int32)

    # [num_fsas, num_fsas, num_fsas, ... ]
    den_graphs_indexes = torch.tensor([num_fsas] * num_fsas, dtype=torch.int32)

    # [0, num_fsas, 1, num_fsas, 2, num_fsas, ... ]
    num_den_graphs_indexes = torch.stack(
        [num_graphs_indexes, den_graphs_indexes]).t().reshape(-1).to(device)

    num_den_reordered_graphs = k2.index(num_den_graphs, num_den_graphs_indexes)

    # [[0, 1, 2, ...]]
    a_to_b_map = torch.arange(num_fsas, dtype=torch.int32).reshape(1, -1)

    # [[0, 1, 2, ...]] -> [0, 0, 1, 1, 2, 2, ... ]
    a_to_b_map = a_to_b_map.repeat(2, 1).t().reshape(-1).to(device)

    num_den_lats = k2.intersect_dense(num_den_reordered_graphs,
                                      dense_fsa_vec,
                                      output_beam=10.0,
                                      a_to_b_map=a_to_b_map)

    num_den_tot_scores = num_den_lats.get_tot_scores(log_semiring=True,
                                                     use_double_scores=True)

    num_tot_scores = num_den_tot_scores[::2]
    den_tot_scores = num_den_tot_scores[1::2]

    tot_scores = num_tot_scores - den_scale * den_tot_scores
    tot_score, tot_frames, all_frames = get_tot_objf_and_num_frames(
        tot_scores, supervision_segments[:, 2])
    return tot_score, tot_frames, all_frames


def _compute_mmi_loss_exact_non_optimized(
        nnet_output: torch.Tensor,
        texts: List[str],
        supervision_segments: torch.Tensor,
        graph_compiler: MmiTrainingGraphCompiler,
        P: k2.Fsa,
        den_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    See :func:`_compute_mmi_loss_exact_optimized` for the meaning
    of the arguments.

    It's more readable, though it invokes k2.intersect_dense twice.

    Note:
      It uses less memory at the cost of speed. It is slower.
    '''
    num_graphs, den_graphs = graph_compiler.compile(texts,
                                                    P,
                                                    replicate_den=True)
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=10.0)
    den_lats = k2.intersect_dense(den_graphs, dense_fsa_vec, output_beam=10.0)

    num_tot_scores = num_lats.get_tot_scores(log_semiring=True,
                                             use_double_scores=True)

    den_tot_scores = den_lats.get_tot_scores(log_semiring=True,
                                             use_double_scores=True)

    tot_scores = num_tot_scores - den_scale * den_tot_scores
    tot_score, tot_frames, all_frames = get_tot_objf_and_num_frames(
        tot_scores, supervision_segments[:, 2])
    return tot_score, tot_frames, all_frames


def _compute_mmi_loss_pruned(
        nnet_output: torch.Tensor,
        texts: List[str],
        supervision_segments: torch.Tensor,
        graph_compiler: MmiTrainingGraphCompiler,
        P: k2.Fsa,
        den_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    See :func:`_compute_mmi_loss_exact_optimized` for the meaning
    of the arguments.

    `pruned` means it uses k2.intersect_dense_pruned

    Note:
      It uses the least amount of memory, but the loss is not exact due
      to pruning.
    '''
    num_graphs, den_graphs = graph_compiler.compile(texts,
                                                    P,
                                                    replicate_den=False)

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, output_beam=10.0)

    # the values for search_beam/output_beam/min_active_states/max_active_states
    # are not tuned. You may want to tune them.
    # 20 7 30 10000
    # for wsj: 10 5 30 10000
    # for aishell 20 7 30 10000
    den_lats = k2.intersect_dense_pruned(den_graphs,
                                         dense_fsa_vec,
                                         search_beam=10.0,
                                         output_beam=5.0,
                                         min_active_states=30,
                                         max_active_states=10000)

    num_tot_scores = num_lats.get_tot_scores(log_semiring=True,
                                             use_double_scores=True)

    den_tot_scores = den_lats.get_tot_scores(log_semiring=True,
                                             use_double_scores=True)

    tot_scores = num_tot_scores - den_scale * den_tot_scores
    tot_score, tot_frames, all_frames = get_tot_objf_and_num_frames(
        tot_scores, supervision_segments[:, 2])
    return tot_score, tot_frames, all_frames


class LFMMILoss(nn.Module):
    """
    Computes Lattice-Free Maximum Mutual Information (LFMMI) loss.

    TODO: more detailed description
    """

    def __init__(
            self,
            graph_compiler: MmiTrainingGraphCompiler,
            P: k2.Fsa,
            use_pruned_intersect: bool = False,
            den_scale: float = 1.0,
    ):
        super().__init__()
        self.graph_compiler = graph_compiler
        self.P = P
        self.den_scale = den_scale
        self.use_pruned_intersect = use_pruned_intersect

    def forward(self, nnet_output: torch.Tensor, texts: List[str],
                supervision_segments: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_pruned_intersect:
            func = _compute_mmi_loss_pruned
        else:
            #  func = _compute_mmi_loss_exact_non_optimized
            func = _compute_mmi_loss_exact_optimized

        return func(nnet_output=nnet_output,
                    texts=texts,
                    supervision_segments=supervision_segments,
                    graph_compiler=self.graph_compiler,
                    P=self.P,
                    den_scale=self.den_scale)

# Author: Jinchuan Tian ; Jan 2022
# jinchuantian@stu.pku.edu.cn

# We test our code on k2 version 1.2; other versions may encounter problems due to API change.
# This file contains the MMI-related utility functions:
# 1. The (equivalent implementation of) step composition between the training / decoding graph;
# 2. The Lattice generation process with look-ahead mechanism.

from typing import List
from typing import Optional
from typing import Tuple

import torch
import k2
import _k2
import numpy as np # debug
from k2 import Fsa, DenseFsaVec 

"""
Intersection function without autograd.

(1) We write this function since the arc_map_a is not accessible in k2 API
(2) Currently we are not using the pruned version to keep all paths.
    We will try to find a balance between the speed and the precision later.
"""
def intersect_dense_forward(a_fsas: Fsa,
                           b_fsas: DenseFsaVec,
                           search_beam: float,
                           output_beam: float,
                           prune: bool,
                           min_active_states: int,
                           max_active_states: int,
                           seqframe_idx_name: Optional[str] = None,
                           frame_idx_name: Optional[str] = None): 

    out_fsa = [0]

    if prune:
        ragged_arc, arc_map_a, arc_map_b = _k2.intersect_dense_pruned(
            a_fsas=a_fsas.arcs,
            b_fsas=b_fsas.dense_fsa_vec,
            search_beam=search_beam,
            output_beam=output_beam,
            min_active_states=min_active_states,
            max_active_states=max_active_states)
    else:
        ragged_arc, arc_map_a, arc_map_b = _k2.intersect_dense(
            a_fsas=a_fsas.arcs,
            b_fsas=b_fsas.dense_fsa_vec,
            a_to_b_map=None,
            output_beam=output_beam)

    out_fsa[0] = Fsa(ragged_arc)

    seqframe_idx = None
    if frame_idx_name is not None:
        num_cols = b_fsas.dense_fsa_vec.scores_dim1()
        seqframe_idx = arc_map_b // num_cols
        shape = b_fsas.dense_fsa_vec.shape()
        fsa_idx0 = _k2.index_select(shape.row_ids(1), seqframe_idx)
        frame_idx = seqframe_idx - _k2.index_select(
            shape.row_splits(1), fsa_idx0)
        assert not hasattr(out_fsa[0], frame_idx_name)
        setattr(out_fsa[0], frame_idx_name, frame_idx)

    if seqframe_idx_name is not None:
        if seqframe_idx is None:
            num_cols = b_fsas.dense_fsa_vec.scores_dim1()
            seqframe_idx = arc_map_b // num_cols

        assert not hasattr(out_fsa[0], seqframe_idx_name)
        setattr(out_fsa[0], seqframe_idx_name, seqframe_idx)

    return out_fsa[0], arc_map_a, arc_map_b


# For each state, Add the score on the ending arc if the end state is reachable 
# from this state. Then return the frame-level scores for each Fsa.
def step_trace(out_fsas, a_fsas, arc_map_a, durations):
    assert out_fsas.shape[0] == a_fsas.shape[0]
    num_fsa = a_fsas.shape[0]

    # K2 FsaVec Meta-info: num_state; 0; 
    # state_accumulated_counts (row_splits1); 
    # arc_accumulated_counts (row_splits12);
    
    # 1.1 Find all a_fsas arcs and meta-info
    a_fsa_dict = a_fsas.as_dict()
    a_fsa_meta = a_fsa_dict["arcs"][: 2 * num_fsa + 4].long()
    a_fsa_arcs = a_fsa_dict["arcs"][2 * num_fsa + 4:].view(-1, 4) 

    # 1.2 Assign global state-ids
    for i in range(num_fsa):
        a_fsa_arcs[a_fsa_meta[i+num_fsa+3]: a_fsa_meta[i+num_fsa+4]][:, :2] += a_fsa_meta[i + 2]

    # 1.3 Find all ending states and their scores. -1 means arcs entering ending states.
    a_fsa_ending_mask = a_fsa_arcs[:, 2] == -1
    a_ending_states = torch.masked_select(a_fsa_arcs[:, 0], a_fsa_ending_mask)
    a_ending_scores = torch.masked_select(a_fsas.scores, a_fsa_ending_mask)

    # 2.1 Find all out_fsas arcs and sort by entering states 
    out_fsa_dict = out_fsas.as_dict()
    out_fsa_meta = out_fsa_dict["arcs"][:2 * num_fsa + 4].long()
    out_fsa_arcs = out_fsa_dict["arcs"][2 * num_fsa + 4:].view(-1, 4)
    out_incoming_ragged = out_fsas._get_incoming_arcs()

    # 2.2 For each state, find an arc entering it
    #     No entering arcs for start states but a fake one is assigned to it.
    #     We need the corresponding arcs in a_fsas so arc_map_a is selected
    transform_index = out_incoming_ragged.values().long()
    select_index = out_incoming_ragged.row_splits(2).long()[:-1]
    arc_map_a_uniq = arc_map_a[transform_index][select_index]
    frame_idx = out_fsas.frame_idx[transform_index][select_index]

    # 2.3 Find all corresponding arcs in a_fsas and their entering states
    #     Starting states of each Fsa is set to 0 to disable the fake arcs
    #     They are not needed to be accurate as long as 0 is not in `a_ending_states`
    a_fsa_arcs_uniq = a_fsa_arcs[arc_map_a_uniq.long()]
    a_states_uniq = a_fsa_arcs_uniq[:, 1]
    # We use this to avoid out-of-range error when last several FSAs are empty
    start_indices = torch.where(out_fsa_meta[2: 2 + num_fsa] == out_fsa_meta[2 + num_fsa],
                                0, out_fsa_meta[2: 2 + num_fsa])
    a_states_uniq[start_indices] = 0

    # 3.1 Find the forward scores
    #     Add ending state scores to the raw state_scores 
    #     if the final state is reachable. Else set to -inf
    raw_state_scores = out_fsas._get_forward_scores(True, True)
    state_scores = torch.ones_like(raw_state_scores) * float('-1e10')
    for state, score in zip(a_ending_states, a_ending_scores):
        state_scores = torch.where(a_states_uniq==state, 
                                   raw_state_scores + score, 
                                   state_scores)
    
    # 3.2 Allocate scores on each frames and each Fsa
    #     Score on starting state is also accumulated on frame 1
    #     But it is always ok since this score is always -inf
    #     TODO: maybe we need assume T in dense_fsa_vec supervison is identical 
    frame_ids, counts = torch.unique_consecutive(frame_idx, return_counts=True)

    score_sequences, start = [], 0
    score_sequence = []
    for i, (fid, fc) in enumerate(zip(frame_ids.tolist(), counts.tolist())):
        frame_score = torch.logsumexp(state_scores[start: start+fc], dim=0)        
        score_sequence.append(frame_score)
        start += fc

        if i == len(counts) - 1 or fid > frame_ids[i+1]:
            score_sequences.append(torch.stack(score_sequence, dim=0)[:-1])
            score_sequence = []

    # 3.3 For empty Fsas, assgin -inf score sequences.
    ans, index = [], 0
    is_empties = out_fsa_meta[2: 2 + num_fsa] == out_fsa_meta[3: 3 + num_fsa]
    for i, is_empty in enumerate(is_empties):
        if is_empty:
            ans.append(torch.ones(durations[i]) * float('-1e10'))
        else:
            ans.append(score_sequences[index])
            index += 1
    assert len(ans) == num_fsa
    return ans

"""
Step intersection implementation

Input:
fsa, FsaVec, training graph like CTC, MMI. Need duplication.
dense_fsa_vec, DenseFsaVec, created from nnet_output and the corresponding length in t-axis.
prune: bool, If true, use a pruned version of intersection.
search_beam: float, parameter used in pruned intersection only.
output_beam: float, paramtere used in intersection.
min_active_states: int, parameter used in pruned intersection only.
max_active_states: int, parameter used in pruned intersection only.

Output: 
score_sequences: List of 1-D tensors. The number of tensors is equal to the number fsas in of `fsa`
                 Each tensor has length of T where T is the number of effective frames in nnet_ouptut.
                 The t-th element represent the `tot_score` of interseted Fsa beteewn the input `fsa` 
                 and the first t frames.

This implementation is much faster than using a loop for T times. As the intersection is only used once
for each Fsa. The sequence is recovered from the output Fsa and the arc_map_a generated by T frames.

The intersection would fail if the hypothesis is too long to reach the end-state. If all intersection
fails, the step_trace would result in an error so we check the out_fsa in advance. (Currently we only
use this in MMI prefix score computation so all hypotheses and dense_fsa_vec have the same lengths. 
This could also be used in MMI alignment score computation with time-axis lookahead and this function
might be further revised later)

Currently we find the `search_beam` and the `output_beam` should be very large to achieve accurate
decoding performance.
"""
def step_intersect(fsa, 
                   dense_fsa_vec, 
                   prune=False, 
                   search_beam=3000, 
                   output_beam=2000,
                   min_active_states=30,
                   max_active_states=5000000):

    out_fsa, arc_map_a, arc_map_b = intersect_dense_forward(
      a_fsas = fsa,
      b_fsas = dense_fsa_vec,
      search_beam = search_beam,
      output_beam = output_beam,
      prune = prune,
      min_active_states = min_active_states,
      max_active_states = max_active_states,
      seqframe_idx_name = "seqframe_idx",
      frame_idx_name = "frame_idx"
    )

     # If all intersections fail
    if out_fsa.num_arcs == 0:
        ans = []
        for d in dense_fsa_vec.duration:
            ans.append(torch.ones(d.item()) * float("-1e10"))
        return ans

    return step_trace(out_fsa, fsa, arc_map_a, dense_fsa_vec.duration) 

def step_intersect_test(): 
    from pathlib import Path
    lang=Path("data/lang_phone")
    device = torch.device("cpu")
    
    # import for test only
    from espnet.nets.scorer_interface import PartialScorerInterface
    from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
    from snowfall.lexicon import Lexicon
    from snowfall.training.mmi_graph import create_bigram_phone_lm

    lexicon = Lexicon(lang)
    oov = open(lang / 'oov.txt').read().strip()
    graph_compiler = MmiTrainingGraphCompiler(lexicon, device, oov)
    phone_ids = lexicon.phone_symbols()

    torch.manual_seed(888)
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.randn_like(P.scores)

    texts = ["你", "好"]
    # texts = ['本 市 警 察 近', '本 市 警 察 今', '本 市 警 察 信', '本 市 警 察 昨', '本 市 警 察 二', '本 市 警 察 记', '本 市 警 察 数', '本 市 警 察 甚', '本 市 警 察 继', '本 市 警 察 至', '本 市 警 察 进', '本 市 警 察 几', '本 市 警 察 五', '本 市 警 察 仅', '本 市 警 察 上']
    num, den = graph_compiler.compile(texts, P, replicate_den=True)
    graph = num 
 
    T = 3
    beam_size = len(texts)
    odim = len(phone_ids) + 1
    nnet_output = torch.rand([beam_size, T, odim])

    supervision = torch.stack([
                          torch.arange(beam_size),
                          torch.zeros(beam_size),
                          torch.ones(beam_size) * T,
                          ], dim=-1).cpu().int()   
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision) 
    score_sequences = step_intersect(graph, 
                                    dense_fsa_vec,
                                    prune=False,
                                    search_beam=30,
                                    output_beam=20,
                                    min_active_states=30,
                                    max_active_states=100000) 
    print(score_sequences)
    return
    print("####  old method ###")
    buf = []
    for t in range(1, T+1):
        supervision = torch.stack([
                          torch.arange(beam_size),
                          torch.zeros(beam_size),
                          torch.ones(beam_size) * t,
                          ], dim=-1).cpu().int()
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        num_lats = k2.intersect_dense(graph, dense_fsa_vec, output_beam=30.0)
        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)
        buf.append(num_tot_scores)

    buf = torch.stack(buf, dim=1)
    score_sequences = torch.stack(score_sequences, dim=0)
    print(buf - score_sequences)
 
if __name__ == "__main__":
    step_intersect_test() 

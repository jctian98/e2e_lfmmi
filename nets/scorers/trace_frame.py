import torch
import k2
import numpy as np
import _k2

"""
def _trace_frame(lats): 
    arcs = lats[0].as_dict()['arcs']
    lats[0].draw("den_3frame.svg")

    frame2state = []
    prev_buf, cur_buf = [0], []

    for arc in arcs:
        f, t, _, _ = arc
        f, t = int(f), int(t)

        if f in prev_buf:
            if not t in cur_buf:
                cur_buf.append(t)

        else:
            frame2state.append(prev_buf)
            prev_buf = cur_buf
            cur_buf = [t]
    
    frame2state.append(prev_buf) # last frame
    frame2state.append([t]) # final state
    return frame2state
"""

def trace_lattice(lats):
    arcs = lats.arcs.values()[:, :2]
    T = max(lats.frame).item()
    frame2state = [[] for _ in range(T+1)]

    for idx, (_, dst) in enumerate(arcs.tolist()):
        frame_idx = lats.frame[idx]
        if dst not in frame2state[frame_idx]:
            frame2state[frame_idx].append(dst)
     
    return frame2state

def compute_frame_level_scores(graph, nnet_output):
    T = nnet_output.size()[1]

    # dump lattice
    supervision = torch.Tensor([[0, 0, T]]).to(torch.int32) 
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
    lats = k2.intersect_dense(graph, dense_fsa_vec, output_beam=10.0,\
           seqframe_idx_name='seqframe', frame_idx_name='frame')
    
    # compute frame-level scores
    forward_scores = lats.get_forward_scores(True, True)
    frame2states = trace_lattice(lats)
    assert len(frame2states) == T + 1 # extra final state

    tot_scores = []
    for t in range(T, 0, -1):
        # scores for the last frame
        if t == T:
            tot_scores.append(forward_scores[-1])
        
        # scores for other frames
        else:
            states = frame2states[t-1]
            frame_score = torch.logsumexp(forward_scores[states], dim=-1)
            tot_scores.append(frame_score)
    tot_scores = torch.stack(tot_scores, dim=0)
    
    return tot_scores

def trace_lattice_batch(lats, batch):
    T = max(lats.frame).item()
    frame2state = [[[] for _ in range(T+1)] for __ in range(batch)] # 2-D list: [batch, T]
    arcs = lats.arcs.values()[:, :2].tolist() 

    batch_idx, last_is_zero = -1, False
    for idx, (src, dst) in enumerate(arcs):
        
        if src == 0 and last_is_zero == False:
            batch_idx += 1
            last_is_zero = True

        if not src == 0:
            last_is_zero = False

        frame_idx = lats.frame[idx]
        if dst not in frame2state[batch_idx][frame_idx]:
            frame2state[batch_idx][frame_idx].append(dst) 

    return frame2state

def split_forward_scores(scores):
    # splits the forward_scores according to the start state
    scores_splits = []
    prev_idx = 0 
    for i in range(1, len(scores)):
        if scores[i] == 0:
            scores_splits.append(scores[prev_idx: i])
            prev_idx = i
    scores_splits.append(scores[prev_idx:])
    return scores_splits

def compute_frame_level_scores_batch(graphs, nnet_output):
    # We would assume that nnet_output in different batch
    # is the same. This is only used for batch decoding
    batch, T, _ = nnet_output.size()

    supervision = torch.stack([
                  torch.arange(batch),
                  torch.zeros(batch),
                  torch.ones(batch) * T
                  ], dim=1).to(torch.int32)
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
    lats = k2.intersect_dense(graphs, dense_fsa_vec, output_beam=10.0,
              seqframe_idx_name='seqframe', frame_idx_name='frame')

    forward_scores = lats.get_forward_scores(True, True)
    forward_scores = split_forward_scores(forward_scores)

    frame2state = trace_lattice_batch(lats, batch)
    
    tot_scores = [[] for _ in range(batch)]
    for b in range(batch):
        for f in range(T, 0, -1): # descent order
            state = frame2state[b][f-1]
            frame_score = torch.logsumexp(forward_scores[b][state], dim=-1)
            tot_scores[b].append(frame_score)
    tot_scores = torch.Tensor(tot_scores)
    return tot_scores 

# this only for debug
def compute_frame_level_scores_loop(graph, nnet_output):
    T = nnet_output.size()[1]

    tot_scores = []
    for t in range(T, 0, -1):
        # feed one more frame it it is not the last frame
        # so the states in first t frames is identical to
        # the those in whole lattice
        t_ = t if t == T else t + 1
        supervision = torch.Tensor([[0, 0, t_]]).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        lats = k2.intersect_dense(graph, dense_fsa_vec, output_beam=10.0,\
               seqframe_idx_name='seqframe', frame_idx_name='frame')

        forward_scores = lats.get_forward_scores(True, True)
        frame2states = trace_lattice(lats)
        
        if t == T:
            tot_scores.append(forward_scores[-1])
        else:
            assert len(frame2states) == t + 2
            states = frame2states[t-1]
            frame_score = torch.logsumexp(forward_scores[states], dim=-1) 
            tot_scores.append(frame_score)
    tot_scores = torch.stack(tot_scores, dim=0)
    return tot_scores

if __name__ == '__main__':
    batch_size = 3
    nnet_output = torch.tensor(
    [
     [0.1, 0.22, 0.28, 0.4],
     [0.1, 0.13, 0.07, 0.7],
     [0.6, 0.2, 0.05, 0.15],
    ], dtype=torch.float32
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    nnet_output = torch.nn.functional.log_softmax(nnet_output, -1)

    graph = k2.ctc_graph([[1], [1,2], [1,2,3]])
    
    scores = compute_frame_level_scores_batch(graph, nnet_output)
    
    #scores = compute_frame_level_scores(graph, nnet_output)
    #print("Scores computed by new version: ", scores)

    #scores = compute_frame_level_scores_loop(graph, nnet_output)
    #print("Scores computed by original version: ", scores)

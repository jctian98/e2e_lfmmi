import torch 

def build_word_mapping(word_mapping):
    ans = {}
    for line in open(word_mapping):
        f, t = line.split()
        ans[int(f)] = int(t)
    return ans 

def convert_transcription(ys, mapping, words, oov_id, ignore_ids):
    """
    ys: 2-D torch tensor. indexs of tokens
    mapping: dict, from attention domain to MMI domain. No special tokens
    words: dict, from MMI domain index to words
    ignore_ids: list, ids to ignore
    
    We assume there should be NO KEY ERROR!
    """
    ys = ys.cpu().numpy()
    ys = [
          [mapping.get(tok, oov_id) for tok in y if not tok in ignore_ids]
          for y in ys
         ]
    ys = [
          " ".join([words[tok] for tok in y])
          for y in ys
         ]
    return ys

def encode_supervision(hlens):
    batch_size = hlens.size()[0]
    supervision = torch.stack((torch.arange(batch_size),
                              torch.zeros(batch_size),
                              hlens.cpu()), 1).to(torch.int32)
    supervision = torch.clamp(supervision, min=0)
    indices = torch.argsort(supervision[:, 2], descending=True)
    supervision = supervision[indices]
    return supervision, indices

def parse_step(hyp, words, part_ids, weights, full_scores, part_scores, weighted_scores):
    # previous hypothesis
    word_hypo = "".join([words[x] for x in hyp.yseq])
    print(f"Previous Hypothesis:   {word_hypo}")
    print(f"Previous Total scores: {hyp.score}")
    
    # candidates:
    part_toks = "     ".join([words[tok] for tok in part_ids])
    print(f"Proposed Candidates:   {part_toks}")

    # slice full scores by part_ids. 
    # cannot modify the original data 
    weighted_scores_sliced = weighted_scores[part_ids]
    full_scores_sliced = {}
    for k in full_scores:
        full_scores_sliced[k] = full_scores[k][part_ids]

    # show scores from every source
    score_dict = {**full_scores_sliced, **part_scores}
    for k in score_dict:
        info = "{:<7}(weighted):   ".format(k)
        for v in score_dict[k]:
            info += "{:>6.2f} ".format(v * weights[k])
        print(info, flush=True)

    score_dict = {**full_scores_sliced, **part_scores, "total": weighted_scores_sliced}
    for k in score_dict:
        info = "{:<7}:             ".format(k)
        for v in score_dict[k]:
            info += "{:>6.2f} ".format(v)
        print(info, flush=True)

import k2
import torch
import numpy as np

def search_lexical_tree(self, node, next_tokens):
    if node is None:
        print("None node given!")

    intervals, next_nodes = [], []
    # some tokens are invalid (e.g., invalid word combination). however, we should
    # still compute the score for it to be compatible. We will force that score
    # to logzero in postprocess stage, but need to use the index_to_kill make a record.
    index_to_kill = []

    for idx, i in enumerate(next_tokens):
        # node is the previous one if _ is not proposed else root
        subword = self.char_list[i]
        # case (1): '_' or <eos> is proposed, which means end of the word
        if subword == self.bpe_space or subword == "<eos>":
            this_node = node # keep 'node' unchanged
            # Invalid and kill. Previous node cannot be root
            if this_node == self.lexroot:
                interval = [self.word_unk_id-1, self.word_unk_id]
                this_node = None
                index_to_kill.append(idx)
            # score is for a word, not a word prefix -> interval for only one word 
            else:
                interval = [this_node[2][0], this_node[2][0] + 1]
                # next_node is root so the next token is valid even though it is not
                # start with '_' 
                this_node = self.lexroot

        # case (2): impossible token. kill them
        elif subword == "<blank>" or subword == "<unk>":
            this_node = None
            interval = [self.word_unk_id-1, self.word_unk_id]
            index_to_kill.append(idx)

        # case (3): ordinary tokens. All special token should never reach this branch
        else:
            # subword start with '_' means a prefix of new word -> search from root
            this_node = self.lexroot if subword.startswith(self.bpe_space) else node

            subword = subword.replace(self.bpe_space, "")
            for c in subword:
                cid = self.alphabet_dict[c]
                # descent to successor
                if cid in this_node[0]:
                    this_node = this_node[0][cid]
                # no valid successor found. kill this hypothesis
                else:
                    this_node = None
                    break

            if this_node is not None and this_node[2] is not None:
                interval = this_node[2]
            else:
                interval = [self.word_unk_id-1, self.word_unk_id]
                index_to_kill.append(idx)

        # plus one to correct the interval. see building process of lexroot
        interval = [interval[0] + 1, interval[1] + 1]
        intervals.append(interval)
        # this_node == None always means a kill
        next_nodes.append(this_node)

    return intervals, next_nodes, index_to_kill


def parse_lookahead(yseq, lexroot, char_list, alphabet, word_dict, bpe_space):

    # (1) check if the final word finishes
    final_token = char_list[yseq[-1]]
    if final_token in ["<blank>", "<eos>", "<unk>", bpe_space]:
        tail_complete = True
    else:
        tail_complete = False

    # (2) recover the string
    yseq = "".join([char_list[y] for y in yseq])\
           .replace("<blank>", "")\
           .replace("<eos>", "")\
           .replace("<unk>", bpe_space + "<unk>")\
           .replace(bpe_space, " ")\
           .strip().split()

    # (3) parse prefix
    unk_id = word_dict["<UNK>"]
    prefix = [word_dict[tok] if tok in word_dict else unk_id 
                for tok in yseq[:-1]]

    # (4) parse interval of tail

    tail = yseq[-1] if len(yseq) > 0 else "<unk>"
    if tail == "<unk>":
        interval = [unk_id-1, unk_id]
    else:
        node = lexroot
        for c in tail:
            cid = alphabet[c]
            if cid in node[0]:
                node = node[0][cid]
                interval = [node[2][0], node[2][0] + 1]\
                               if tail_complete else node[2]
            else:
                interval = [unk_id-1, unk_id]
                break

    # shift by 1: see building process of lexroot
    interval = [interval[0] + 1, interval[1] + 1]   

    # yseq = " ".join(yseq)
    # print(f"yseq: {yseq} prefix: {prefix} interval: {interval}") 
    return prefix, interval

def build_word_fsa_mat(prefix, interval):
    prefix_len = len(prefix)

    # prefix part
    start_state = np.arange(prefix_len)
    end_state = np.arange(prefix_len) + 1
    labels = np.array(prefix)
    scores = np.zeros(prefix_len)
    prefix_part = np.stack([start_state, end_state, labels, scores], axis=1)

    # interval_part
    interval_len = interval[1] - interval[0]
    start_state = np.ones(interval_len) * prefix_len
    end_state = np.ones(interval_len) * (prefix_len + 1)
    labels = np.arange(interval[0], interval[1])
    scores = np.zeros(interval_len)
    interval_part = np.stack([start_state, end_state, labels, scores], axis=1)

    # final arc
    final_arc = np.array([[prefix_len + 1, prefix_len + 2, -1, 0]])

    # combine 
    mat = np.concatenate([prefix_part, interval_part, final_arc], axis=0)
    mat = torch.from_numpy(mat).int()
    return mat

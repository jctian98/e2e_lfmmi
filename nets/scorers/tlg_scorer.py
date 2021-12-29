# Author: tyriontian
# tyriontian@tencent.com

import os
import sys
import torch
import kaldi.fstext as fst

from pathlib import Path
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.nets.scorers.sorted_matcher import SortedMatcher

class TlgPartialScorer(PartialScorerInterface):
    """
    This is a wrapper for Espnet: the word-level N-gram LM on-the-fly decoding method.
    (proposed by cweng, cweng@tencent.com)
    """

    def __init__(self, lang, nonblk_reward=0.0):
        self.lang = Path(lang)
        
        # build the SortedMatcher: core of this algorithm
        # the `lang` directory should have these files 
        disambig_ids = open(self.lang / "disambig_ids").readline().replace("\n", "").split(",")
        disambig_ids = [int(i) for i in disambig_ids]
        backoff_id = int(open(self.lang / "backoff_id").readline().strip())
        max_id = int(open(self.lang / "max_id").readline().strip())
        max_num_arcs = int(open(self.lang / "max_num_arcs").readline().strip())
        fst_lm = fst.StdVectorFst.read(str(self.lang / "LG.fst"))

        self.scorer = SortedMatcher(fst_lm, max_num_arcs, max_id, backoff_id, disambig_ids) 
        
        # reward whenever a new non-blank token generated
        assert nonblk_reward >= 0.0
        self.nonblk_reward = nonblk_reward

        print("Build TLG scorer successfully!", flush=True)

    def init_state(self, x=None):
        """
        0 is the starting state
        """
        return {0: 0.0}

    def score_partial(self, y, next_tokens, state, x):
        """
        args:
        y: interface required. Not used here
        next_tokens: list of token-ids to search
        state: dict, {state1: score1, state2: score2, ...}
               state is shared for all token-ids
        x: interface required, Not used here

        return:
        scores: list of scores for each token-id
        next_states: list of dicts, each of which is in format like `state`

        Hint: next_tokens contains no <blank> 
        """
        scores = []
        next_states = []
        for tok_id in next_tokens:
            # <eps> is not in our vocab but in the compilation of LG.fst
            score, next_state = self.score_one(tok_id + 1, state)
            scores.append(score)
            next_states.append(next_state)

        return scores, next_states

    def score_one(self, tok_id, state_dict):
        # In case the searched results are all empty.
        scores = [1e10]
        next_states = [0]
        for state, prev_score in state_dict.items():
            searched = list(self.scorer.get_scores(state, tok_id))
            searched[0] = [x + prev_score for x in searched[0]]
            scores += searched[0]
            next_states += searched[1]
        
        # the scores used for comparison have considered previous scores.  
        next_dict = {}
        for state, score in zip(next_states, scores):
            if state in next_dict:
                next_dict[state] = min(next_dict[state], score)
            else:
                next_dict[state] = score
        
        next_dict = {k: v + self.nonblk_reward for k, v in next_dict.items()}
        # Minimum value in the state dict is exactly the accumulated socre of the 
        # whole history. The first-order difference is the token-level score.
        score = min(next_dict.values()) - min(state_dict.values())
        return - score, next_dict
           
    def final_score(self, states):
        """
        args:
        states: list of dict {state1: score1, state2: score2, ...}
        
        return: 
        scores: final scores for each hypothesis
        state are not returned and considered any longer
        """
        scores = []
        for state in states:
            score = self.final_score_one(state)
            scores.append(score)
        return scores

    def final_score_one(self, state_dict):
        scores = []
        for state, _ in state_dict.items():
            searched = self.scorer.final_score(state)
            scores += searched[0]
        score = min(scores) - min(state_dict.values())
        return score
        
if __name__ == "__main__":
   token_list = [s.split()[0] for s in open("data/char.txt").readlines()]
   token_list.insert(0, "<blk>")
   scorer = TlgPartialScorer("data/tlg_ngram", token_list=token_list) 

   texts = ["天空很蓝", "天坑很蓝", "我爱你", "我艾你", "宇智波鼬", "宇子波鼬", "翁超","余剑威", "田晋川"]
   for text in texts:
       text_ids = [token_list.index(t) for t in text]
       state = scorer.init_state(None)
       for text_id in text_ids:
           score, next_states = scorer.score_partial(None, [text_id], state, None)
           state = next_states[0]
           print(f"token: {token_list[text_id]} | score: {score} | state: {state}")
       score = scorer.final_score([state])
       print(f"Final score: {score}") 
   

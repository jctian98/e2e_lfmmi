"""Search algorithms for transducer models."""
from typing import List
from typing import Union
from collections import Counter, defaultdict
import numpy as np
import torch
import time
import math
from itertools import groupby
from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import init_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract
from espnet.nets.transducer_decoder_interface import Hypothesis
from espnet.nets.transducer_decoder_interface import NSCHypothesis
from espnet.nets.transducer_decoder_interface import TransducerDecoderInterface
from espnet.nets.scorers.mmi_rescorer import MMIRescorer
# from espnet.nets.scorers.mmi_rnnt_scorer import MMIRNNTScorer
from espnet.nets.scorers.mmi_alignment_score import MMIRNNTScorer
from espnet.nets.scorers.ctc_rnnt_scorer import CTCRNNTScorer
from espnet.nets.scorers.mmi_rnnt_lookahead_scorer import MMIRNNTLookaheadScorer

class BeamSearchTransducer:
    """Beam search implementation for transducer."""

    def __init__(
        self,
        decoder: Union[TransducerDecoderInterface, torch.nn.Module],
        joint_network: torch.nn.Module,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        char_list = None,
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        score_norm: bool = True,
        nbest: int = 1,
        mmi_scorer=None,
        mmi_weight=0.0,
        ctc_module=None,
        ctc_weight=0.0,
        ngram_scorer=None,
        ngram_weight=0.0,
        word_ngram_scorer=None,
        word_ngram_weight=0.0,
        tlg_scorer=None,
        tlg_weight=0.0,
        forbid_eng=False,
        eng_vocab=None,
    ):
        """Initialize transducer beam search.

        Args:
            decoder: Decoder class to use
            joint_network: Joint Network class
            beam_size: Number of hypotheses kept during search
            lm: LM class to use
            lm_weight: lm weight for soft fusion
            search_type: type of algorithm to use for search
            max_sym_exp: number of maximum symbol expansions at each time step ("tsd")
            u_max: maximum output sequence length ("alsd")
            nstep: number of maximum expansion steps at each time step ("nsc")
            prefix_alpha: maximum prefix length in prefix search ("nsc")
            score_norm: normalize final scores by length ("default")
            nbest: number of returned final hypothesis
        """
        self.decoder = decoder
        self.joint_network = joint_network

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim
        self.blank = decoder.blank

        # MMI alignment scorer
        self.mmi_scorer = mmi_scorer
        self.mmi_weight = mmi_weight
        print(f"MMI scorer: {mmi_scorer} | MMI weight: {mmi_weight}")

        # deprecated. CTC scorer
        self.ctc_module = ctc_module
        self.ctc_weight = ctc_weight
        print(f"CTC scorer: {ctc_module} | CTC weight: {ctc_weight}")

        # character-level Ngram scorer implemented by kenlm
        self.ngram_scorer = ngram_scorer
        self.ngram_weight = ngram_weight
        print(f"ngram scorer: {ngram_scorer} | ngram weight: {ngram_weight}")

        # word-level Ngram scorer implemented by k2
        self.word_ngram_scorer = word_ngram_scorer
        self.word_ngram_weight = word_ngram_weight
        print(f"word ngram scorer: {word_ngram_scorer} | word ngram weight: {word_ngram_weight}")

        # word-level Ngram scorer implemented by pykaldi (cweng)
        self.tlg_scorer = tlg_scorer
        self.tlg_weight = tlg_weight
        print(f"tlg scorer: {tlg_scorer} | tlg weight: {tlg_weight}")

        if search_type == "ctc_greedy":
            self.search_algorithm = self.ctc_greedy_search
            assert self.ctc_module is not None
        elif self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "ctc_beam":
            self.search_algorithm = self.ctc_beam_search
            assert self.ctc_module is not None
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            self.search_algorithm = self.nsc_beam_search
        else:
            raise NotImplementedError

        self.lm = lm
        self.lm_weight = lm_weight
        print(f"Using LM {lm} with weight {lm_weight}")

        if lm is not None and lm_weight > 0.0:
            self.use_lm = True
            self.is_wordlm = True if hasattr(lm, "predictor") and \
                             hasattr(lm.predictor, "wordlm") else False
            if hasattr(lm, "predictor"):
                self.lm_predictor = lm.predictor.wordlm if self.is_wordlm else lm.predictor
                self.lm_layers = len(self.lm_predictor.rnn)
            else:
                self.is_transformer_lm = True
        else:
            self.use_lm = False

        self.max_sym_exp = max_sym_exp
        self.u_max = u_max
        self.nstep = nstep
        self.prefix_alpha = prefix_alpha
        self.score_norm = score_norm

        self.nbest = nbest
        self.char_list = char_list

        self.forbid_lst = []
        if forbid_eng:
            self.forbid_lst = [self.char_list.index(x) \
                               for x in self.char_list \
                               if (x >= '\u0041' and x <= '\u005a') \
                               or (x >= '\u0061' and x <= '\u007a')]
        print("Forbid chars: ", self.forbid_lst, flush=True)

        self.eng_vocab = eng_vocab
        if self.eng_vocab is not None:
            self.eng_token_list = [x if not is_all_chinese(x) else "" \
                                   for x in self.char_list]

    def __call__(self, h: torch.Tensor) -> Union[List[Hypothesis], List[NSCHypothesis]]:
        """Perform beam search.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(h.device)

        if len(h.size()) == 3:
            h = h.squeeze(0)

        if not hasattr(self.decoder, "decoders"):
            self.decoder.set_data_type(h.dtype)

        nbest_hyps = self.search_algorithm(h)
        
        if isinstance(self.mmi_scorer, MMIRescorer):
            nbest_hyps = self.mmi_scorer.score(h, nbest_hyps, v2=True)
        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[NSCHypothesis]]
    ) -> Union[List[Hypothesis], List[NSCHypothesis]]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: list of hypotheses

        Return:
            hyps: sorted list of hypotheses

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def vocab_regularization(self, hyps):
        bpe_seperator = u'\u2581'
 
        ans = []
        for h in hyps:
            yseq = h.yseq if isinstance(h, Hypothesis) else h[0] # rnnt or ctc hypothesis
            text = "".join([self.eng_token_list[x] for x in yseq[1:]])
            eng_words = [x for x in text.split(bpe_seperator)[:-1] if x != ""] # the last may not finish
            if all([x in self.eng_vocab for x in eng_words]):
                ans.append(h)
 
        return ans

    def greedy_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation for transformer-transducer.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            hyp: 1-best decoding results

        """
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)
        cache = {}

        y, state, _ = self.decoder.score(hyp, cache)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.joint_network(hi, y), dim=-1)
            logp, pred = torch.max(ytu, dim=-1)
            if pred != self.blank:
                hyp.yseq.append(int(pred))
                hyp.score += float(logp)

                hyp.dec_state = state

                y, state, _ = self.decoder.score(hyp, cache)
        return [hyp]

    def ctc_greedy_search(self, h: torch.Tensor) -> List[Hypothesis]:
        if len(h.size()) == 2:
            h = h.unsqueeze(0)       
 
        lpz = self.ctc_module.argmax(h)
        collapsed_indices = [x[0] for x in groupby(lpz[0])]
        hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
        nbest_hyps = [Hypothesis(score=0.0, yseq=[self.blank] + hyp, dec_state=None)]
        return nbest_hyps

    # mainly derived from wenet
    def ctc_beam_search(self, h: torch.Tensor) -> List[Hypothesis]:
        if len(h.size()) == 2:
            h = h.unsqueeze(0)

        ctc_prob = self.ctc_module.log_softmax(h)[0]
        maxlen = ctc_prob.size(0)

        use_full_score = False
        if self.word_ngram_weight > 0.0:
            lm, lm_weight = self.word_ngram_scorer, self.word_ngram_weight
        elif self.ngram_weight > 0.0:
            lm, lm_weight = self.ngram_scorer, self.ngram_weight
        elif self.lm_weight > 0.0:
            lm, lm_weight = self.lm, self.lm_weight
            use_full_score = True
        else:
            lm, lm_weight = None, 0.0
        
        if lm is not None:
            # yseq: (lm_score, lm_state)
            lm_cache = {(self.blank,): (0.0, lm.init_state(None))}
            sort_fn = lambda x: log_add(list(x[1])) + lm_cache[x[0]][0]
        else:
            lm_cache = None
            sort_fn = lambda x: log_add(list(x[1])) 

        # non-blank sequence; (blank_ending_score, non_blank_ending_score)
        cur_hyps = [((self.blank,), (0.0, -float('inf')))]
        for t in range(0, maxlen):
            logp = ctc_prob[t]
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            top_k_logp, top_k_index = logp.topk(self.beam_size)
     
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()

                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == self.blank: # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        #  Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # LM on-the-fly rescore for unseen prefix
            if lm is not None:
                for prefix, (_, _) in next_hyps.items():
                    if not prefix in lm_cache.keys():
                        y = prefix[:-1]
                        # update all children hypotheses: NNLM 
                        if use_full_score:
                            scores, state = lm.score(torch.Tensor(y).long(), 
                                                     lm_cache[y][1], h)
                            for k in range(len(scores)):
                                lm_cache[y + (k,)] = (lm_cache[y][0] \
                                  + scores[k].item() * lm_weight, 
                                  lm.select_state(state, k)
                                )
                        # update only this hypothesis: N-gram LM                                
                        else:
                            next_token = prefix[-1:]
                            score, state = lm.score_partial(
                                             torch.Tensor(y).long(), 
                                             torch.Tensor(next_token).long(),
                                             lm_cache[y][1], h
                                           )
                            lm_cache[prefix] = (lm_cache[y][0] + score[0].item() * lm_weight, 
                                                lm.select_state(state, 0)
                                               )
             
            next_hyps = sorted(next_hyps.items(), key=sort_fn, reverse=True)
            if self.eng_vocab:
                next_hyps = self.vocab_regularization(next_hyps)
            cur_hyps = next_hyps[:self.beam_size]

        hyps = [Hypothesis(score=log_add([hyp[1][0], hyp[1][1]]),
                           yseq=list(hyp[0]),
                           dec_state=None,
                           mmi_tot_score=lm_cache[hyp[0]][0] \
                             if lm_cache is not None else 0.0
                           )
                           for hyp in cur_hyps
               ]

        return hyps 

    def default_beam_search(self, h: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Args:
            x: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)]
        cache = {}

        for hi in h:
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                y, state, lm_tokens = self.decoder.score(max_hyp, cache)

                ytu = torch.log_softmax(self.joint_network(hi, y), dim=-1)
                top_k = ytu[1:].topk(beam_k, dim=-1)
                
                # add a blank only
                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(ytu[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    lm_state, lm_scores = self.lm.predict(max_hyp.lm_state, lm_tokens)
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[0][k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        cache = {}

        if self.use_lm and not self.is_wordlm:
            B[0].lm_state = init_lm_state(self.lm_predictor)

        for hi in h:
            A = []
            C = B

            h_enc = hi.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                        )

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_states = create_lm_batch_state(
                            [c.lm_state for c in C], self.lm_layers, self.is_wordlm
                        )

                        beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                            beam_lm_states, beam_lm_tokens, len(C)
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                                new_hyp.lm_state = select_lm_state(
                                    beam_lm_states, i, self.lm_layers, self.is_wordlm
                                )

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, h: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """

        hidden = h
        beam = min(self.beam_size, self.vocab_size)

        h_length = int(h.size(0))
        u_max = min(self.u_max, (h_length - 1))

        beam_state = self.decoder.init_state(beam)        
 
        B = [
            Hypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
                mmi_tot_score=0.0,
                word_ngram_score=0.0,
                tlg_state=self.tlg_scorer.init_state() if self.tlg_scorer else None
            )
        ]
        # final hypothesis set to return
        final = []
        # For hypothesis with same yseq, its decoder output could be cached
        # yseq -> decoder_out, decoder_state
        cache = {} 

        # lm initialization
        if self.use_lm and not self.is_wordlm:
            if hasattr(self, "lm_predictor"):
                B[0].lm_state = init_lm_state(self.lm_predictor)
            else:
                B[0].lm_state = self.lm.init_state(h)

        if self.mmi_scorer is not None and self.mmi_weight > 0.0:
            mmi_nnet_output, mmi_den_scores = self.mmi_scorer.den_scores(h)

        if self.ctc_module:
            ctc_pred = self.ctc_module.log_softmax(h.unsqueeze(0))[0]

        for tu_sum in range(h_length + u_max):
            A = [] # collection for next step
            B_ = [] # collection for search in this step. state of pred. net is kept
            h_states = [] # collection of encoder_out frame for each hypothesis
            p_ctc = [] # collection of ctc distribution
            for j, hyp in enumerate(B): # skip all hypothesis that head the last frame.
                u = len(hyp.yseq) - 1
                t = tu_sum - u + 1

                if t > (h_length - 1):
                    continue

                B_.append(hyp)
                h_states.append((t, h[t]))

                if self.ctc_module:
                    p_ctc.append(ctc_pred[t])

            if B_:
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_,
                    beam_state,
                    cache,
                    self.use_lm,
                )
                
                h_enc = torch.stack([h[1] for h in h_states])

                # [beam, h_dim], [beam, h_dim]
                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1) # [beam, vocab]
                if self.forbid_lst:
                    beam_logp[:, self.forbid_lst] = -1e20
                    beam_logp = torch.log_softmax(beam_logp, dim=-1)              
 
                if self.ctc_module:
                    p_ctc = torch.stack([p for p in p_ctc])
                    beam_logp += self.ctc_weight * p_ctc

                # warning: like in LASCTC, the LM score would not be considered in top-k process
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1) # values and indices: [beam, beam]. blank excluded

                if self.use_lm and not self.is_transformer_lm:
                    beam_lm_states = create_lm_batch_state(
                        [b.lm_state for b in B_], self.lm_layers, self.is_wordlm
                    )

                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(B_)
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        mmi_tot_score=hyp.mmi_tot_score,
                        word_ngram_score=hyp.word_ngram_score,
                        tlg_state=hyp.tlg_state,
                    )

                    if h_states[i][0] == (h_length - 1):
                        final.append(new_hyp)
                    
                    A.append(new_hyp)

                    # Only search a part of candidate tokens
                    if self.word_ngram_scorer and self.word_ngram_weight > 0.0:
                        next_tokens = beam_topk[1][i] + 1
                        word_ngram_scores, word_ngram_states = self.word_ngram_scorer.score_partial(
                                                                 hyp.yseq[1:], next_tokens, 
                                                                 hyp.word_ngram_score, None)
                    else:
                        word_ngram_scores = [0.0] * len(beam_topk[1][i])
                        word_ngram_states = [0.0] * len(beam_topk[1][i])

                    if self.tlg_scorer and self.tlg_weight > 0.0:
                        next_tokens = beam_topk[1][i] + 1
                        tlg_scores, tlg_states = self.tlg_scorer.score_partial(
                                                     None, next_tokens,
                                                     hyp.tlg_state, None)
                    else:
                        tlg_scores = [0.0] * len(beam_topk[1][i])
                        tlg_states = [None] * len(beam_topk[1][i])

                    if self.use_lm and self.is_transformer_lm:
                        lm_score, lm_state = self.lm.score(torch.Tensor(hyp.yseq).long(),
                                                           hyp.lm_state,
                                                           None)
                     
                    for j, (logp, k) in enumerate(zip(beam_topk[0][i], beam_topk[1][i] + 1)):
 
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                            mmi_tot_score=hyp.mmi_tot_score,
                            word_ngram_score=word_ngram_states[j],
                            tlg_state=tlg_states[j] 
                        )

                        # add LM scores. possibly 5 styles
                        if self.use_lm and not self.is_transformer_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]

                            new_hyp.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )

                        if self.use_lm and self.is_transformer_lm:
                            new_hyp.score += self.lm_weight * lm_score[k]

                            new_hyp.lm_state = lm_state   
 
                        # Word-level N-gram LM
                        if self.word_ngram_scorer and self.word_ngram_weight > 0.0:
                            new_hyp.score += self.word_ngram_weight * word_ngram_scores[j]

                        # TLG.fst
                        if self.tlg_scorer and self.tlg_weight > 0.0:
                            new_hyp.score += self.tlg_weight * tlg_scores[j]

                        # N-gram LM
                        if self.ngram_scorer and self.ngram_weight > 0.0:
                            ngram_score, _ = self.ngram_scorer.score_partial(
                                             torch.Tensor(hyp.yseq[:]).int(),
                                             torch.Tensor([int(k)]).int(), 
                                             None, h)
                            new_hyp.score += self.ngram_weight * ngram_score.item()
                            
                        A.append(new_hyp)
           
            if self.eng_vocab is not None:
                A = self.vocab_regularization(A)
 
            if self.mmi_scorer is not None and self.mmi_weight > 0.0:
                A = self.mmi_scorer.batch_score(A, mmi_nnet_output, mmi_den_scores, tu_sum+1, self.mmi_weight)

            # unlike the original implementation, we combine the hypotheses before pruning
            # this allow the hypothesis different and possibly make the rescore more effective
            B = recombine_hyps(A, self.mmi_weight)
            B = sorted(B, key=lambda x: x.score, reverse=True)[:beam]
 
        if self.tlg_scorer and self.tlg_weight > 0.0 and final:
            tlg_final_states = [h.tlg_state for h in final]
            tlg_final_scores = self.tlg_scorer.final_score(tlg_final_states)
            for i, h in enumerate(final):
                h.score += self.tlg_weight * tlg_final_scores[i]

        if self.mmi_scorer is not None and self.mmi_weight == 0.0:
            final = self.mmi_scorer.batch_rescore(final, hidden)

        if final:
            return self.sort_nbest(final)
        else:
            print("Warning: No finished hypothesis found. return the partial hypothesis", flush=True)
            return B

    def nsc_beam_search(self, h: torch.Tensor) -> List[NSCHypothesis]:
        """N-step constrained beam search implementation.

        Based and modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Note: the algorithm is not in his "complete" form but works almost as
        intended.

        Args:
            h: Encoded speech features (T_max, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            NSCHypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                None, beam_lm_tokens, 1
            )
            lm_state = select_lm_state(
                beam_lm_states, 0, self.lm_layers, self.is_wordlm
            )
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            NSCHypothesis(
                yseq=[self.blank],
                score=0.0,
                dec_state=state,
                y=[beam_y[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for hi in h:
            hyps = sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True)
            kept_hyps = []

            h_enc = hi.unsqueeze(0)

            for j, hyp_j in enumerate(hyps[:-1]):
                for hyp_i in hyps[(j + 1) :]:
                    curr_id = len(hyp_j.yseq)
                    next_id = len(hyp_i.yseq)

                    if (
                        is_prefix(hyp_j.yseq, hyp_i.yseq)
                        and (curr_id - next_id) <= self.prefix_alpha
                    ):
                        ytu = torch.log_softmax(
                            self.joint_network(hi, hyp_i.y[-1]), dim=-1
                        )

                        curr_score = hyp_i.score + float(ytu[hyp_j.yseq[next_id]])

                        for k in range(next_id, (curr_id - 1)):
                            ytu = torch.log_softmax(
                                self.joint_network(hi, hyp_j.y[k]), dim=-1
                            )

                            curr_score += float(ytu[hyp_j.yseq[k + 1]])

                        hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

            S = []
            V = []
            for n in range(self.nstep):
                beam_y = torch.stack([hyp.y[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(self.joint_network(h_enc, beam_y), dim=-1)
                beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

                for i, hyp in enumerate(hyps):
                    S.append(
                        NSCHypothesis(
                            yseq=hyp.yseq[:],
                            score=hyp.score + float(beam_logp[i, 0:1]),
                            y=hyp.y[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )
                    )

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        score = hyp.score + float(logp)

                        if self.use_lm:
                            score += self.lm_weight * float(hyp.lm_scores[k])

                        V.append(
                            NSCHypothesis(
                                yseq=hyp.yseq[:] + [int(k)],
                                score=score,
                                y=hyp.y[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                                lm_scores=hyp.lm_scores,
                            )
                        )

                V.sort(key=lambda x: x.score, reverse=True)
                V = substract(V, hyps)[:beam]

                beam_state = self.decoder.create_batch_states(
                    beam_state,
                    [v.dec_state for v in V],
                    [v.yseq for v in V],
                )
                beam_y, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                if self.use_lm:
                    beam_lm_states = create_lm_batch_state(
                        [v.lm_state for v in V], self.lm_layers, self.is_wordlm
                    )
                    beam_lm_states, beam_lm_scores = self.lm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(V)
                    )

                if n < (self.nstep - 1):
                    for i, v in enumerate(V):
                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            v.lm_scores = beam_lm_scores[i]

                    hyps = V[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(h_enc, beam_y), dim=-1
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.y.append(beam_y[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = select_lm_state(
                                beam_lm_states, i, self.lm_layers, self.is_wordlm
                            )
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)

# wenet log_add implementation used in beam search
def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

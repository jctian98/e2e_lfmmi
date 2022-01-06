"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging
import os
import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.nets.scorers.mmi_lookahead import MMILookaheadScorer
from espnet.nets.scorers.mmi_frame_scorer import MMIFrameScorer
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.mmi_rescorer import MMIRescorer
from espnet.nets.scorers.word_ngram import WordNgramPartialScorer
from espnet.utils.rtf_calculator import RTF_calculator

def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details

    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = torch.device("cuda")
    else:
        # So the cuda is not available now
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        assert torch.cuda.is_available() == False
    print(f"Rank: {args.local_rank} Using device: {device}, ngpu: {args.ngpu}")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.rnnlm, lm)
        lm.eval()
    else:
        lm = None

    if args.ngram_model and args.ngram_weight > 0.0:
        from espnet.nets.scorers.ngram import NgramFullScorer
        from espnet.nets.scorers.ngram import NgramPartScorer

        if args.ngram_scorer == "full":
            ngram = NgramFullScorer(args.ngram_model, train_args.char_list)
        else:
            ngram = NgramPartScorer(args.ngram_model, train_args.char_list)
    else:
        ngram = None

    # load mmi_scorer
    if args.mmi_weight > 0.0:
        # Also make sure it is K2MMI
        assert hasattr(model.ctc, "dump_weight")
        # Dump a pth for each rank to avoid conflits when reading / writing
        model.ctc.dump_weight(args.local_rank)
        print(f"Using MMI scorer type: {args.mmi_type}")
        mmi_scorer = MMIFrameScorer if args.mmi_type == "frame" else MMILookaheadScorer
        mmi = mmi_scorer(lang=model.ctc.lang,
                         device=device,
                         idim=train_args.adim,
                         sos_id=model.sos,
                         rank=args.local_rank,
                         use_segment=args.use_segment,
                         char_list=train_args.char_list)
    else:
        mmi = None

    if args.mmi_rescore:
        model.ctc.dump_weight(args.local_rank)
        assert args.mmi_weight <= 0.0
        mmi_rescorer = MMIRescorer(lang=model.ctc.lang,
                                   device=device,
                                   idim=train_args.adim,
                                   sos_id=model.sos,
                                   rank=args.local_rank,
                                   use_segment=args.use_segment,
                                   char_list=train_args.char_list)
    else:
        mmi_rescorer = None

    if args.ctc_weight > 0.0:
        ctc_module = model.third_loss if hasattr(model, "third_loss") else model.ctc
        ctc = CTCPrefixScorer(ctc_module, model.eos)
    else: 
        ctc = None

    if args.word_ngram_weight > 0.0:
        word_ngram_scorer = WordNgramPartialScorer
        print(f"Using word ngram model: {args.word_ngram}", flush=True)
        word_ngram_scorer = WordNgramPartialScorer(args.word_ngram, 
                              device,
                              train_args.char_list, 
                              log_semiring=args.word_ngram_log_semiring)
    else:
        word_ngram_scorer = None
        
    scorers = model.scorers()
    scorers["ctc"] = ctc 
    scorers["mmi"] = mmi 
    scorers["lm"] = lm
    scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    scorers["word_ngram"] = word_ngram_scorer
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty,
        mmi=args.mmi_weight,
        word_ngram=args.word_ngram_weight,
    )
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(train_args.char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=train_args.char_list,
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "full",
        mmi_rescorer=mmi_rescorer,
    )
    # TODO(karita): make all scorers batchfied
    if args.batchsize == 1:
        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            beam_search.__class__ = BatchBeamSearch
            logging.info("BatchBeamSearch implementation is selected.")
        else:
            logging.warning(
                f"As non-batch scorers {non_batch} are found, "
                f"fall back to non-batch implementation."
            )

    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()
    beam_search.to(device=device, dtype=dtype).eval()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}
    rtf_calculator = RTF_calculator(js)
    rtf_calculator.tik()
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            enc = model.encode(torch.as_tensor(feat).to(device=device, dtype=dtype))
            nbest_hyps = beam_search(
                x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
            )
            nbest_hyps = [
                h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), args.nbest)]
            ]
            new_js[name] = add_results_to_json(
                js[name], nbest_hyps, train_args.char_list
            )
    
    rtf_calculator.tok()    

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )

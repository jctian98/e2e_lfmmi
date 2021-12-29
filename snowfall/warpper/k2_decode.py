import torch
import k2
import sys
import numpy as np
import logging

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from kaldialign import edit_distance

MAX_LEN = 2000

def k2_decode(model, device, js, sampler, batch_size, use_segment=False):
    model.ctc.decode_init()
    model.to(device)
    model.eval()

    egs = []
    tot_results = []
    tot_loss = []
    num_egs = len(list(js.keys()))
    for idx, name in enumerate(js.keys()):
        egs.append((name, js[name]))
   
        if len(egs) == batch_size or idx == num_egs - 1:
            # for logging
            names = [eg[0] for eg in egs]
            if not use_segment: # chinese
                texts = [eg[1]["output"][0]["token"] for eg in egs] 
            else: # english
                texts = [eg[1]["output"][0]["text"] for eg in egs]
            ilens_from_json = [eg[1]["input"][0]["shape"][0] for eg in egs]
            batch_size = len(names) # for last several examples

            feats = sampler(egs)
            xs_pad, ilens = build_batch_data(feats[0])
            xs_pad = torch.from_numpy(xs_pad).to(device)
            ilens = torch.from_numpy(ilens).to(device)
            egs = []

            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2) 
            hs_pad, hs_mask = model.encoder(xs_pad, src_mask)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
                
            results = model.ctc.decode(hs_pad, hs_len, texts, use_segment) 
            tot_results += results  
            parse_results(tot_results)

def build_batch_data(feats):
    # feats: list of 2d ndarray
    batch_size = len(feats)
    dim = feats[0].shape[-1]
    max_len = 0
    buf = np.zeros((batch_size, MAX_LEN, dim), dtype=np.float32)
    ilen = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        feat = feats[i]
        feat_len = feat.shape[0]
        buf[i:, :feat_len, :] = feat
        ilen[i] = feat_len
        max_len = max(max_len, feat_len)

    buf = buf[:, :max_len, :]
    return buf, ilen

def parse_results(results):
    dists = [edit_distance(r, h) for r, h in results]
    errors = {
        key: sum(dist[key] for dist in dists)
        for key in ['sub', 'ins', 'del', 'total']
    }
    total_chars = sum(len(ref) for ref, _ in results)
    logging.warning(
        f'%WER {errors["total"] / total_chars:.2%} '
        f'[{errors["total"]} / {total_chars}, {errors["ins"]} ins, {errors["del"]} del, {errors["sub"]} sub ]'
    )


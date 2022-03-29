# Author: Jinchuan Tian ; tianjinchuan@stu.pku.edu.cn ; tyriontian@tencent.com
# This script provides:
# (1) CER (2) Bayesian Risk and its variance

import sys
import json
import math
import editdistance
import numpy as np

def main():
    # load json file
    f = open(sys.argv[1], "rb")
    results_json = json.load(f)["utts"]

    num_err, num_tot = 0, 0
    risk_stat, sum_prob_stat, ref_prob_stat = [], [], []
    for uttid, info in results_json.items():
        try:
            hypotheses = info["output"]
            ref_token = hypotheses[0]["token"]
        
            # hypothesis and their probability
            texts, probs, find_ref = [], [], False
            for h in hypotheses:
                text = h["rec_token"].replace("<eos>", "").strip()
                texts.append(text)
                probs.append(math.exp(h["score"]))
                if ref_token == text:
                    ref_prob_stat.append(math.exp(h["score"]))
                    find_ref = True
    
            if not find_ref:
                ref_prob_stat.append(0.0)
    
            # find edit-distance
            edit_dists = [editdistance.eval(ref_token, rec_token) \
                          for rec_token in texts]
    
            # bayesian risk
            weighted_probs = [a * b for a, b in zip(edit_dists, probs)]
            risk = sum(weighted_probs) / (sum(probs) + 1e-10)
            risk_stat.append(risk)
    
            # sum prob 
            sum_prob_stat.append(sum(probs))
    
            # cer statistics
            num_err += edit_dists[0]
            num_tot += len(ref_token.strip().split())
        except:
            pass

    # conclusion
    print("### MBR statistics on {} ###".format(sys.argv[1]))
    cer = num_err / num_tot * 100
    print("CER: {:.4f}% {}/{}".format(cer, num_err, num_tot))
    
    br_mean, br_deviation = np.mean(risk_stat), np.sqrt(np.var(risk_stat))
    print("Mean and Deviation of Bayesian Risk: {:.4f} | {:.4f}".format(br_mean, br_deviation))

    sum_prob_mean, sum_prob_deviation = np.mean(sum_prob_stat), np.sqrt(np.var(sum_prob_stat))
    ref_prob_mean, ref_prob_deviation = np.mean(ref_prob_stat), np.sqrt(np.var(ref_prob_stat))
    print("Mean and Deviation of Accumulated probability: {:.4f} | {:.4f}".format(sum_prob_mean, sum_prob_deviation))
    print("Mean and Deviation of Reference probability: {:.4f} | {:.4f}".format(ref_prob_mean, ref_prob_deviation))


if __name__ == "__main__":
    main() 

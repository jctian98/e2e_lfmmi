# author: tyriontian
# tyriontian@tencent.com

import json
import sys
import codecs
import torch

from espnet.nets.scorers.word_ngram import WordNgram

def score_texts(ngram, texts, ignore_strs=["<eos>", " "]):
    for s in ignore_strs:
        texts = [t.replace(s, "") for t in texts]
    return ngram.score_texts(texts).cpu().tolist()

def main():
    js_file = sys.argv[1]
    ngram_dir = sys.argv[2]
    weight = float(sys.argv[3])
    js_file_tgt = sys.argv[4]

    # read json
    with codecs.open(js_file, "r", encoding="utf-8") as f:
        js = json.load(f)
        js = js["utts"]
    
    # load word-level N-gram LM
    device = torch.device("cpu")
    ngram = WordNgram(ngram_dir, device)

    # rescore each hypothesis and sort
    for name in js.keys():
        hyp_lst = js[name]["output"]

        texts = []
        for hyp in hyp_lst:
            texts.append(hyp["rec_text"])

        text_scores = score_texts(ngram, texts)
        
        for j, hyp in enumerate(hyp_lst):
            hyp_lst[j]["score"] += text_scores[j] * weight
            hyp_lst[j]["word_ngram_score"] = text_scores[j]

        hyp_lst.sort(key=lambda hyp: hyp["score"], reverse=True)
        js[name]["output"] = hyp_lst

    js = {"utts": js}

    # write new json
    with open(js_file_tgt, "wb") as f:
        f.write(
            json.dumps(
                js, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
    
    


if __name__ == "__main__":
    main()

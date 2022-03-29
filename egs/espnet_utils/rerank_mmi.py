import sys
import json
import codecs


json_f = sys.argv[1]
json_f_out = sys.argv[3]
weight = float(sys.argv[2])

with codecs.open(json_f, "r", encoding="utf-8") as f:
        j = json.load(f)

for name in j["utts"]:
    hyp_lst = j["utts"][name]["output"]
    for hyp in hyp_lst:
        hyp["score"] = float(hyp["score"]) + float(hyp["mmi_tot_score"]) * weight
    hyp_lst.sort(key=lambda hyp: hyp["score"], reverse=True)
    j["utts"][name]["output"] = hyp_lst

with open(json_f_out, "wb") as f:
    f.write(
        json.dumps(
            j, indent=4, ensure_ascii=False, sort_keys=True
        ).encode("utf_8")
    )

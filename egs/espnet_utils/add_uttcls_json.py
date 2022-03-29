import json
import sys

def main():
    in_json = sys.argv[1]
    out_json = sys.argv[2]
    clsid = sys.argv[3]

    reader = open(in_json, encoding="utf-8")
    j = json.load(reader)

    for name in j["utts"].keys():
        j["utts"][name]["output"][0]["tokenid"] = \
            clsid + " " + j["utts"][name]["output"][0]["tokenid"]

    with open(out_json, "wb") as f:
        f.write(
            json.dumps(
                j, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )

if __name__ == "__main__":
    main()

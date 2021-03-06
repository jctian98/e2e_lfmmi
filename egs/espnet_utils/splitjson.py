#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import codecs
import json
import logging
import os
import sys

import numpy as np

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="split a json file for parallel processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json", type=str, help="json file")
    parser.add_argument(
        "--parts", "-p", type=int, help="Number of subparts to be prepared", default=0
    )
    parser.add_argument(
        "--original-order", action="store_true", help="If set, not sort utts by keys"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info(get_commandline_args())

    # check directory
    filename = os.path.basename(args.json).split(".")[0]
    dirname = os.path.dirname(args.json)
    dirname = "{}/split{}utt".format(dirname, args.parts)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # load json and split keys
    j = json.load(codecs.open(args.json, "r", encoding="utf-8"))
    if args.original_order:
        utt_ids = list(j["utts"].keys())
    else:
        utt_ids = sorted(list(j["utts"].keys()))
    logging.info("number of utterances = %d" % len(utt_ids))
    if len(utt_ids) < args.parts:
        logging.error("#utterances < #splits. Use smaller split number.")
        sys.exit(1)
    utt_id_lists = np.array_split(utt_ids, args.parts)
    utt_id_lists = [utt_id_list.tolist() for utt_id_list in utt_id_lists]

    for i, utt_id_list in enumerate(utt_id_lists):
        new_dic = dict()
        for utt_id in utt_id_list:
            new_dic[utt_id] = j["utts"][utt_id]
        jsonstring = json.dumps(
            {"utts": new_dic},
            indent=4,
            ensure_ascii=False,
            sort_keys=not args.original_order,
            separators=(",", ": "),
        )
        fl = "{}/{}.{}.json".format(dirname, filename, i + 1)
        sys.stdout = codecs.open(fl, "w+", encoding="utf-8")
        print(jsonstring)
        sys.stdout.close()

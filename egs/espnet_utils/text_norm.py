# author: tyriontian
# tyriontian@tencent.com

import sys
import os
import jieba
import argparse
import cn2an
from string import punctuation as en_pun
from zhon.hanzi import punctuation as zh_pun
pun = en_pun + zh_pun

def remove_punc(s):
    for c in pun:
        s = s.replace(c, "")
    return s

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def splice(s):
    # the segmentation results are only on Chinese
    # we do not splice English
    buf = []
    ans = []
    for c in s:
        if is_all_chinese(c):
            if buf:
                buf_str = "".join(buf)
                buf = []
                ans.append(buf_str)
            ans.append(c)
        else:
            buf.append(c)

    # incase the last c is not Chinese
    if buf:
        buf_str = "".join(buf)
        ans.append(buf_str)

    ans = " ".join(ans)
    return ans

digit_dict = {"0": "零",
              "1": "一",
              "2": "二",
              "3": "三",
              "4": "四",
              "5": "五",
              "6": "六",
              "7": "七",
              "8": "八",
              "9": "九",}
def digit_norm(s):
    out = ""
    buf = ""
    for c in s:
        if not c.isdigit():
            if buf:
                try:
                    digit_str = cn2an.an2cn(buf)
                except:
                    print(f"cannot convert digit {buf}")
                    digit_str = "".join([digit_dict.get(x, "") for x in buf])
                out += digit_str
                buf = ""
            out += c
        else:
            buf += c
    if buf:
        buf = cn2an.an2cn(buf)
        out += buf
    return out
       

def remove_blank_chn(s):
    s = s.strip()
    out = ""
    for i in range(len(s)):
        if not s[i] == " ":
            out += s[i]
        else:
            a = is_all_chinese(s[i-1])
            b = is_all_chinese(s[i+1])
            # if not a and not b: 
            if not a or not b: # keep chn-eng <space> 
                out += s[i]
    return out

def add_blank_boundary(s):
    s = s.strip()
    out = ""
    for i in range(len(s) - 1):
        out += s[i]
        
        a = is_all_chinese(s[i])
        b = is_all_chinese(s[i+1])
        if a ^ b:
            out += " "
    
    out += s[-1]
    out = out.strip()
    return out

def split_eng_words(s):
    out = []
    for w in s.strip().split():
        if is_all_chinese(w):
            out.append(w)
        else:
            for c in w:
                out.append(c)
    out = " ".join(out)
    return out

def split_chn_words(s):
    out = []
    for w in s.strip().split():
        if is_all_chinese(w):
            out.extend(list(w))
        else:
            out.extend(w.split())
    out = " ".join(out)
    return out

def upper_or_lower(s, upper=True):
    if upper:
        return s.upper()
    else:
        return s.lower()
 
def process_one_line(content, args):
    # (1) remove punctuation and space
    content = remove_punc(content)

    # (2) remove ignore symbols
    if args.ignore is not None:
        ignores = args.ignore.split(",")
        for c in ignores:
            content.replace(c, "")
    
    # (3) digit norm and upper/lower
    content = digit_norm(content)
    content = upper_or_lower(content, args.eng_upper)

    # (4) remove all blank except those between eng words
    #     This is for kaldi/text
    content = remove_blank_chn(content)
    
    if args.segment_chn:
        content = split_chn_words(content) 

    if not args.segment:
        return content

    # (5) split by jieba. There should be a blank
    #     at any chn-eng boundary 
    else:
        content = add_blank_boundary(content)
        if args.segment_eng:
            content = split_eng_words(content)
        content = content.strip().split()
        out = []
        for p in content:
            if is_all_chinese(p):
                out.extend(jieba.lcut(p, HMM=False))
            else:
                out.append(p)
        out = " ".join(out)
        return out

def get_parser():
    parser = argparse.ArgumentParser(
        description="Normalize the text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in-f", type=str, help="input file")
    parser.add_argument(
        "--out-f", type=str, help="output file")
    parser.add_argument(
        "--freq-dict", type=str, default=None, help="frequency dict")
    parser.add_argument(
        "--segment", action='store_true', help="do segmentation in output file")
    parser.add_argument(
        "--segment-eng", action='store_true', help="segment english into chars")
    parser.add_argument(
        "--segment-chn", action='store_true', help="segment mandarin into chars")
    parser.add_argument(
        "--ignore", type=str, default=None, help="symbol to remove in output file")
    parser.add_argument(
        "--eng-upper", action='store_true', help="all english in upper class")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.segment and args.freq_dict:
        jieba.set_dictionary(args.freq_dict)

    writer = open(args.out_f, 'w', encoding="utf-8") 
    for line in open(args.in_f, encoding="utf-8"):
        elems = line.strip().split()

        # we skip the empty string
        if len(elems) == 1:
            print(f"Empty text found for {elems[0]}")
            continue

        uttid, content = elems[0], elems[1:]
        content = " ".join(content)

        content = process_one_line(content, args)
        writer.write(f"{uttid} {content}\n")  

if __name__ == "__main__":
    main()

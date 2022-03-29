import sys
import torch
import random
import torchaudio

MAX_LENGTH = 12 * 100 # 12 seconds


def read_datadir(d):
    wav_dict = {}
    for line in open(d + "/wav.scp", encoding="utf-8"):
        uttid, item = line.split()
        wav_dict[uttid] = item

    text_dict = {}
    for line in open(d + "/text", encoding="utf-8"):
        uttid, item = line.split()[0], line.split()[1:]
        text_dict[uttid] = " ".join(item)

    dur_dict = {}
    for line in open(d + "/utt2num_frames", encoding="utf-8"):
        uttid, item = line.strip().split()
        dur_dict[uttid] = int(item)
    
    return wav_dict, text_dict, dur_dict

def generate_pairs(chn_dur_dict, eng_dur_dict, dur_maximum):
    dur_sum = 0 
    ans = []
    chn_keys = list(chn_dur_dict.keys())
    eng_keys = list(eng_dur_dict.keys())
    random.shuffle(chn_keys)
    random.shuffle(eng_keys)

    for i, (chn_k, eng_k) in enumerate(zip(chn_keys, eng_keys)):
        length = chn_dur_dict[chn_k] + eng_dur_dict[eng_k]
        if length > MAX_LENGTH:
            continue

        k_lst = [chn_k, eng_k]
        random.shuffle(k_lst)

        ans.append(k_lst)
        dur_sum += length

        if dur_sum > dur_maximum:
            break

    return ans 

def write_utts(tgt_dir, pair_list, wav_dict, text_dict):
    text_writer = open(tgt_dir + "/text", 'w', encoding="utf-8")
    scp_writer = open(tgt_dir + "/wav.scp", 'w', encoding="utf-8")

    def write_utt(path1, path2, path):
        wave1, sr1 = torchaudio.load(path1)
        wave2, sr2 = torchaudio.load(path2)
        assert sr1 == sr2
        wave = torch.cat([wave1, wave2], dim=-1)
        torchaudio.save(path, wave, sample_rate=sr1)

    for i, (k1, k2) in enumerate(pair_list):
        uttid = k1 + "_and_" + k2
        wave_path = tgt_dir + '/wavs/' + uttid + '.wav'
        write_utt(wav_dict[k1], wav_dict[k2], wave_path)
   
        text = uttid + " " + text_dict[k1] + ' ' + text_dict[k2] + "\n"
        text_writer.write(text)
        text_writer.flush()

        scp_info = uttid + " " + wave_path + "\n"
        scp_writer.write(scp_info)
        scp_writer.flush()

        if i % 10000 == 0:
            print(f"have generate {i} utts")

    text_writer.close()
    scp_writer.close()

def main():

    chn_dir, eng_dir, tgt_dir = sys.argv[1:4]

    chn_wav_dict, chn_text_dict, chn_dur_dict = \
        read_datadir(chn_dir)

    eng_wav_dict, eng_text_dict, eng_dur_dict = \
        read_datadir(eng_dir)
    
    chn_wav_dict.update(eng_wav_dict)
    chn_text_dict.update(eng_text_dict)

    # 200 hours
    pair_list = generate_pairs(chn_dur_dict, eng_dur_dict, dur_maximum=100 * 3600 * 200)
    write_utts(tgt_dir, pair_list, chn_wav_dict, chn_text_dict)

if __name__ == "__main__":
    main()

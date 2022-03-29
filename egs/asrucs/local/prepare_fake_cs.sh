cs_dir=data/train_cs_fake/
# generate wav files
# python3 local/generate_fake_cs.py data/train_zh_trim/ data/train_en_trim/ data/train_cs_fake/

# cat $cs_dir/text | cut -d ' ' -f 1 | awk '{print $1, $1}' > $cs_dir/spk2utt 
# cp $cs_dir/spk2utt $cs_dir/utt2spk
# utils/fix_data_dir.sh $cs_dir

# steps/make_fbank_pitch.sh --nj 500 --write_utt2num_frames true \
#     $cs_dir exp/make_fbank/cs_fake fbank

# dump.sh  --nj 48 --do_delta false \
#     $cs_dir/feats.scp data/cmvn/cmvn.ark exp/dump_feats/fake_cs \
#     dump/train_cs_fake/deltafalse/

# python3 espnet_utils/text_norm.py --in-f data/train_cs_fake/text --out-f data/train_cs_fake/text_org --eng-upper
# mv data/train_cs_fake/text_org data/train_cs_fake/text

# feat_part_dir=dump/train_cs_fake/deltafalse/
# data2json.sh --nj 20 --feat $feat_part_dir/feats.scp --bpecode data/dict_cs/bpe.model \
#     data/train_cs_fake data/dict_cs/dict.txt > $feat_part_dir/data.json

# dict=data/dict_cs/dict.txt
# n_symbols=`wc -l $dict | cut -d ' ' -f 1`

# python3 espnet_utils/add_uttcls_json.py dump/train_cs_fake/deltafalse/data.json \
#             dump/train_cs_fake/deltafalse/data_withcls.json $[$n_symbols + 4]

python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data_withcls.json \
        dump/train_en_trim/deltafalse/data_withcls.json \
        dump/train_cs_fake/deltafalse/data_withcls.json \
        --shuffle > dump/jsons/pretrain_data_fakecs.json
python3 espnet_utils/splitjson.py -p 8 --original-order dump/jsons/pretrain_data_fakecs.json

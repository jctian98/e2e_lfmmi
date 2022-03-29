#!/usr/bin/env bash

# author: tyriontian
# tianjinchuan@stu.pku.edu.cn ; tyriontian@tencent.com

# A Code-Switch ASR recipe

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=2
stop_stage=100
dumpdir=dump
fbankdir=fank
do_delta=false
nbpe=5000
oteam_cs_text=../oteam_asr3/data/eng/text

. utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: prepare features"
    # make raw features and remove long-short utts
    for part in train_zh train_en train_cs; do 
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 500 --write_utt2num_frames true \
            data/$part exp/make_fbank/$part ${fbankdir}
        espnet_utils/remove_longshortdata.sh --maxframes 1200 --maxchars 400 data/${part} data/${part}_trim
    done

    # compute cmvn
    mkdir -p data/cmvn
    cat data/train_zh_trim/feats.scp data/train_en_trim/feats.scp data/train_cs_trim/feats.scp \
        | shuf | head -n 10000 > data/cmvn/feats_cmvn.scp
    compute-cmvn-stats scp:data/cmvn/feats_cmvn.scp data/cmvn/cmvn.ark

    # dump features without speed perturb
    for part in train_zh_trim train_en_trim train_cs_trim \
                dev_zh dev_en dev_cs test_zh test_en test_cs; do
        feat_part_dir=${dumpdir}/${part}/delta${do_delta}; mkdir -p ${feat_part_dir}
        dump.sh --cmd "$train_cmd" --nj 48 --do_delta ${do_delta} \
            data/${part}/feats.scp data/cmvn/cmvn.ark exp/dump_feats/${part} \
            ${feat_part_dir}
    done
fi

mkdir -p data/dict_cs
dict=data/dict_cs/dict.txt
nlsyms=data/dict_cs/nlsyms.txt
bpemodel=data/dict_cs/bpe
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: prepare dictionary and json file"
    
    # prepare special symbols
    echo "<unk> 1" > $dict; echo "<chn> 2" >> $dict; echo "<eng> 3" >> $dict
    echo "<chn>" > $nlsyms; echo "<eng>" >> $nlsyms

    # chn symbols
    text2token.py -s 1 -n 1 data/train_zh_trim/text | cut -f 2- -d" " | tr " " "\n" \
      | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+3}' >> ${dict}
    nchn_symbols=`wc -l $dict | cut -d ' ' -f 1` 

    # eng symbols and bpe models 
    cat data/train_en_trim/text | cut -d ' ' -f 2- > data/dict_cs/bpe_input.txt
    spm_train --input=data/dict_cs/bpe_input.txt --vocab_size=${nbpe} --model_type=unigram \
        --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece --split-chn < data/dict_cs/bpe_input.txt \
        | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+'${nchn_symbols}'}' >> ${dict}

    # eng words
    cat data/train_en_trim/text data/train_cs_trim/text 

    # make json files
    for part in train_zh_trim train_en_trim train_cs_trim \
                dev_zh dev_en dev_cs test_zh test_en test_cs; do
        feat_part_dir=${dumpdir}/${part}/delta${do_delta}
        data2json.sh --nj 20 --feat $feat_part_dir/feats.scp --bpecode ${bpemodel}.model \
            data/$part $dict > $feat_part_dir/data.json 
    done

    # Add language-id in label sequence; consider <blk> <eos>
    # We add these label only for model inference -> no test sets
    n_symbols=`wc -l $dict | cut -d ' ' -f 1`
    for part in train_zh_trim dev_zh; do
        python3 espnet_utils/add_uttcls_json.py dump/${part}/deltafalse/data.json \
            dump/${part}/deltafalse/data_withcls.json $[$n_symbols + 2]
    done

    for part in train_en_trim dev_en; do
        python3 espnet_utils/add_uttcls_json.py dump/${part}/deltafalse/data.json \
            dump/${part}/deltafalse/data_withcls.json $[$n_symbols + 3]
    done

    for part in train_cs_trim dev_cs; do
        python3 espnet_utils/add_uttcls_json.py dump/${part}/deltafalse/data.json \
            dump/${part}/deltafalse/data_withcls.json $[$n_symbols + 4]
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Process json files in multiple styles"
    # dev
    python3 espnet_utils/concatjson.py \
        dump/dev_zh/deltafalse/data.json \
        dump/dev_en/deltafalse/data.json \
        dump/dev_cs/deltafalse/data.json \
        > dump/jsons/dev.json

    # zh + en 
    python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data.json \
        dump/train_en_trim/deltafalse/data.json \
        --shuffle > dump/jsons/zh_en.json
    
    # zh + en + cs
    python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data.json \
        dump/train_en_trim/deltafalse/data.json \
        dump/train_cs_trim/deltafalse/data.json \
        --shuffle > dump/jsons/zh_en_cs.json
    
    # zh + en + fakecs
    python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data.json \
        dump/train_en_trim/deltafalse/data.json \
        dump/train_cs_fake/deltafalse/data.json \
        --shuffle > dump/jsons/zh_en_fakecs.json

    ### With class label ###
    
    # dev
    python3 espnet_utils/concatjson.py \
        dump/dev_zh/deltafalse/data_withcls.json \
        dump/dev_en/deltafalse/data_withcls.json \
        dump/dev_cs/deltafalse/data_withcls.json \
        > dump/jsons/dev_withcls.json

    # zh + en 
    python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data_withcls.json \
        dump/train_en_trim/deltafalse/data_withcls.json \
        --shuffle > dump/jsons/zh_en_withcls.json

    # zh + en + cs
    python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data_withcls.json \
        dump/train_en_trim/deltafalse/data_withcls.json \
        dump/train_cs_trim/deltafalse/data_withcls.json \
        --shuffle > dump/jsons/zh_en_cs_withcls.json

    # zh + en + fakecs
    python3 espnet_utils/concatjson.py \
        dump/train_zh_trim/deltafalse/data_withcls.json \
        dump/train_en_trim/deltafalse/data_withcls.json \
        dump/train_cs_fake/deltafalse/data_withcls.json \
        --shuffle > dump/jsons/zh_en_fakecs_withcls.json

    for json in zh_en zh_en_withcls zh_en_cs zh_en_cs_withcls zh_en_fakecs zh_en_fakecs_withcls; do
        python3 espnet_utils/splitjson.py -p 8 --original-order dump/jsons/${json}.json &
    done; wait
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: build LMs"
    
#    mkdir -p data/ngram
#    # process corpus
#    for part in train_zh train_en train_cs dev_zh dev_en dev_cs; do
#        cut -d " " -f 2- data/$part/text | spm_encode --model=${bpemodel}.model \
#          --output_format=piece > data/ngram/${part}_input.txt
#    done
#    cut -d " " -f 2- $oteam_cs_text | spm_encode --model=${bpemodel}.model \
#          --output_format=piece > data/ngram/oteam_cs_input.txt
#
#    rm -f data/ngram/train_combine_input.txt data/ngram/dev_combine_input.txt 
#    cat data/ngram/train*_input.txt > data/ngram/train_combine_input.txt
#    cat data/ngram/dev*_input.txt   > data/ngram/dev_combine_input.txt
#  
#    # train N-gram LM 
#    for part in zh en cs combine; do 
#        lmplz --discount_fallback -o 5 < data/ngram/train_${part}_input.txt \
#          > data/ngram/${part}_5gram.arpa
#        build_binary data/ngram/${part}_5gram.arpa data/ngram/${part}_5gram.bin
#
#        cat data/ngram/train_${part}_input.txt data/ngram/oteam_cs_input.txt \
#          > data/ngram/train_${part}_input_extended.txt
#        lmplz --discount_fallback -o 5 <  data/ngram/train_${part}_input_extended.txt\
#          > data/ngram/${part}_5gram_oteam.arpa
#        build_binary data/ngram/${part}_5gram_oteam.arpa data/ngram/${part}_5gram_oteam.bin
#    done
#
#    # train token-level LM
#    for part in combine; do
#        ${cuda_cmd} --gpu 4 exp/train_nnlm_${part}/train.log \
#            lm_train.py \
#            --config conf/lm_transformer.yaml \
#            --ngpu 4 \
#            --backend pytorch \
#            --verbose 1 \
#            --outdir exp/train_nnlm_${part} \
#            --train-label data/ngram/train_${part}_input.txt \
#            --valid-label data/ngram/dev_${part}_input.txt \
#            --dict ${dict}
#    done

#    mkdir -p data/word_lm; rm data/word_lm/*
#    for part in train_zh train_en train_cs dev_zh dev_en dev_cs; do
#        python3 espnet_utils/text_norm.py --in-f data/${part}/text \
#          --out-f data/word_lm/${part}_input.txt --eng-upper --segment-chn 
#    done
#    rm -f data/word_lm/train_combine_input.txt 
#    rm -f data/word_lm/dev_combine_input.txt
#    cat data/word_lm/train_* > data/word_lm/train_combine_input.txt
#    cat data/word_lm/dev_*   > data/word_lm/dev_combine_input.txt
#
#    word_dict=data/word_lm/dict.txt
#    text2vocabulary.py -s 65000 -o ${word_dict} data/word_lm/train_combine_input.txt

    
fi
exit 0;

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: build word N-gram LMs"
    mkdir -p data/word_ngram;
   
    # prepare corpus 
    for part in train_zh train_en train_cs; do
        python3 espnet_utils/text_norm.py --in-f data/${part}/text \
          --out-f data/word_ngram/${part}_seg.txt --segment --eng-upper
        cut -d ' ' -f 2- data/word_ngram/${part}_seg.txt | spm_encode \
          --model=${bpemodel}.model --output_format=piece \
          > data/word_ngram/${part}_bpe.txt
        python3 local/add_seperator.py data/word_ngram/${part}_seg.txt \
          data/word_ngram/${part}_word.txt
        cat data/word_ngram/${part}_bpe.txt data/word_ngram/${part}_word.txt \
          > data/word_ngram/${part}_input.txt
    done 
    cat data/word_ngram/*_input.txt > data/word_ngram/train_combine_input.txt

    # prepare words.txt
    words=data/word_ngram/words.txt
    cat data/word_ngram/train_combine_input.txt | tr " " "\n" | sort | uniq | \
      grep -v '<unk>' > ${words} 

    # train N-gram and convert to torch version
    words_disambig=data/word_ngram/words_disambig.txt
    (echo "<eps>"; echo "<unk>") | cat - $words |\
      awk '{print $0 " " NR-1}' > $words_disambig
    echo "#0 `wc -l $words_disambig | cut -d ' ' -f 1`" \
      >> $words_disambig
    for part in train_zh train_en train_cs train_combine; do
        bash espnet_utils/train_lms_srilm.sh \
          --unk "<unk>" --lm-opts -wbdiscount --order 5 \
          $words data/word_ngram/${part}_input.txt \
          data/word_ngram/$part
        gunzip -c data/word_ngram/$part/srilm/srilm.o3g.kn.gz \
          > data/word_ngram/$part/lm.arpa
        cp $words_disambig data/word_ngram/$part/words.txt
        echo 1 > data/word_ngram/$part/oov.int

        python3 -m kaldilm \
          --read-symbol-table=$words_disambig \
          --disambig-symbol='#0' \
          --max-order=5 \
          data/word_ngram/$part/lm.arpa \
          > data/word_ngram/$part/G.fst.txt

        python3 espnet/nets/scorers/word_ngram.py data/word_ngram/$part 
    done 
fi

#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=2         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=             # tag for managing LMs
n_gram=4
# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data dir, modify this to your AISHELL-2 data path
tr_dir=/data/asr_data/aishell2/iOS/data
dev_tst_dir=/data/asr_data/aishell2/AISHELL-DEV-TEST-SET
word_arpa=/apdcephfs/share_1149801/speech_user/tomasyu/jinchuan/data/ngram/cweng_3g_5gram.arpa
# exp tag
### Configurable parameters ###
tag="8v100_rnnt_mmi_ctc"
ngpu=8

# Train config
seed=888
batch_size=8
accum_grad=1
epochs=100
use_segment=true # if true, use word-level transcription in MMI criterion
aux_ctc_weight=0.5
aux_ctc_dropout_rate=0.1
aux_mmi=true
aux_mmi_weight=0.5
aux_mmi_dropout_rate=0.1
aux_mmi_type='mmi' # mmi or phonectc

# Decode config
idx_average=91_100
search_type="alsd" # "default", "nsc", "tsd", "alsd"
mmi_weight=0.2 # MMI / phonectc joint decoding
ctc_weight=0.0 # char ctc joint decoding
ngram_weight=0.0
word_ngram_weight=0.0
word_ngram_order=4 # 3 or 4 gram
mmi_type="frame" # or rescore
beam_size=10
recog_set="test_android test_ios test_mic"

. utils/parse_options.sh || exit 1;

#if [ $debug -eq true ]; then
#    export HOST_GPU_NUM=1
#    export HOST_NUM=1
#    export NODE_NUM=1
#    export INDEX=0
#    export CHIEF_IP="9.135.217.29"
#fi

train_opts=\
"\
--seed $seed \
--batch-size $batch_size \
--accum-grad $accum_grad \
--epochs $epochs \
--use-segment $use_segment \
--aux-ctc $aux_ctc \
--aux-ctc-weight $aux_ctc_weight \
--aux-ctc-dropout-rate $aux_ctc_dropout_rate \
--aux-mmi $aux_mmi \
--aux-mmi-weight $aux_mmi_weight \
--aux-mmi-dropout-rate $aux_mmi_dropout_rate \
--aux-mmi-type $aux_mmi_type \
"

decode_opts=\
"\
--search-type $search_type \
--mmi-weight $mmi_weight \
--beam-size $beam_size \
--ctc-weight $ctc_weight \
--mmi-type $mmi_type \
--ngram-weight $ngram_weight \
--word-ngram-weight $word_ngram_weight \
--word-ngram data/word_${word_ngram_order}gram \
"

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp
train_dev=dev_ios

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    # For training set
    local/prepare_data.sh ${tr_dir} data/local/train data/train || exit 1;
    # # For dev and test set
    for x in Android iOS Mic; do
        local/prepare_data.sh ${dev_tst_dir}/${x}/dev data/local/dev_${x,,} data/dev_${x,,} || exit 1;
        local/prepare_data.sh ${dev_tst_dir}/${x}/test data/local/test_${x,,} data/test_${x,,} || exit 1;
    done 
    # Normalize text to capital letters
    for x in train dev_android dev_ios dev_mic test_android test_ios test_mic; do
        mv data/${x}/text data/${x}/text_org
        paste <(cut -f 1 data/${x}/text_org) <(cut -f 2 data/${x}/text_org | tr '[:lower:]' '[:upper:]') \
            > data/${x}/text
        rm data/${x}/text_org
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 188 --write_utt2num_frames true \
        data/train exp/make_fbank/train ${fbankdir}
    utils/fix_data_dir.sh data/train
    for x in android ios mic; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
            data/dev_${x} exp/make_fbank/dev_${x} ${fbankdir}
        utils/fix_data_dir.sh data/dev_${x}     
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
            data/test_${x} exp/make_fbank/test_${x} ${fbankdir}
        utils/fix_data_dir.sh data/test_${x}
    done
    
    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 30 --write_utt2num_frames true \
    data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
    utils/fix_data_dir.sh data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    split_dir=$(echo $PWD | awk -F "/" '{print $NF "/" $(NF-1)}')
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/a{11,12,13,14}/${USER}/espnet-data/egs/${split_dir}/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/a{11,12,13,14}/${USER}/espnet-data/egs/${split_dir}/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 100 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
        
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 20 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

lang=data/lang_phone
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: prepare lang and do text segmentation"
    bash local/prepare_dict.sh data/local/dict
    local/k2_prepare_lang.sh --position-dependent-phones false data/local/dict \
      "<UNK>" data/local/lang_tmp_nosp $lang || exit 1

    # use jieba for segmentation. This would take a few minutes
    python3 local/jieba_build_dict.py $lang/words.txt $lang/jieba_dict.txt
    for part in train_sp dev_android dev_ios dev_mic test_android test_ios test_mic; do
        python3 local/jieba_split_text.py data/${part}/text data/${part}/text_orig.scp $lang/jieba_dict.txt
    done

    # word-level N-gram model
    mkdir -p data/local/lm
    awk '{print $1}' data/local/dict/lexicon.txt | sort | uniq | awk '{print $1,99}' \
      > data/local/lm/word_seg_vocab.txt
    python2 local/word_segmentation.py data/local/lm/word_seg_vocab.txt \
      data/local/train/text > data/local/lm/trans.txt

    local/train_lms.sh \
     data/local/dict/lexicon.txt \
     data/local/lm/trans.txt \
     data/local/lm || exit 1;

    for order in 3 4; do
      wngram_dir=data/word_${order}gram; mkdir -p $wngram_dir
      cp $lang/words.txt $wngram_dir
      cp $lang/oov.int $wngram_dir
      gunzip -c data/local/lm/${order}gram-mincount/lm_unpruned.gz \
        > $wngram_dir/lm.arpa
      python3 -m kaldilm \
      --read-symbol-table="$wngram_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=$order \
      $wngram_dir/lm.arpa > $wngram_dir/G.fst.txt
    done

    # prepare very large LM with external resources
    mkdir -p data/word_5gram; wdir=data/word_5gram
    cp $lang/words.txt $wdir/
    cp $lang/oov.int $wdir/

    python3 -m kaldilm \
        --read-symbol-table="$wdir/words.txt" \
        --disambig-symbol='#0' \
        --max-order=5 \
        $word_arpa > $wdir/G.fst.txt
    # built the .pt file
    python3 espnet/nets/scorers/word_ngram.py
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 3: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
                 --text_org data/${train_set}/text_orig.scp \
		 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
                     --text_org data/${rtask}/text_orig.scp \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done   
fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " \
        > ${lmdatadir}/valid.txt

    #${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
    #    lm_train.py \
    #    --config ${lm_config} \
    #    --ngpu $ngpu \
    #    --backend ${backend} \
    #    --verbose ${verbose} \
    #    --outdir ${lmexpdir}/results \
    #    --tensorboard-dir ${lmexpdir}/tensorboard \
    #    --train-label ${lmdatadir}/train.txt \
    #    --valid-label ${lmdatadir}/valid.txt \
    #    --resume ${lm_resume} \
    #    --dict ${dict}

    ngramexpdir=exp/train_ngram
    lmplz --discount_fallback -o ${n_gram} <${lmdatadir}/train.txt > ${ngramexpdir}/${n_gram}gram.arpa
    build_binary -s ${ngramexpdir}/${n_gram}gram.arpa ${ngramexpdir}/${n_gram}gram.bin

fi



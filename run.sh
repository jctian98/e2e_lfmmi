#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/tuning/train_pytorch_conformer_kernel31.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=             # tag for managing LMs

# ngram
ngramtag=
n_gram=4

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
data=/data/asr_data/aishell/
data_url=www.openslr.org/resources/33

################# Tunable Paramters #################
ngpu=8
tag="8v100_lasmmictc_alpha03_ctc03" 
seed=888
ctc_type="k2mmi"
mmi_unit="phone" # char or phone
use_segment=true # if true, use text_org for MMI training
debug=false

# Train config
accum_grad=1
batch_size=8
mtlalpha=0.3
third_weight=0.0 # weight of CTCLoss if the MMILoss is used
epochs=50

# Decode config
idx_average=41_50
ctc_weight=0.5
mmi_weight=0.0
lm_weight=0.0
ngram_weight=0.3
beam_size=10
decode_nj=188 #7176  #192 # 14326
mmi_rescore=false
recog_set="test"

. utils/parse_options.sh || exit 1;

train_opts=\
"\
--seed $seed \
--ctc_type $ctc_type \
--accum-grad $accum_grad \
--batch-size $batch_size \
--mtlalpha $mtlalpha  \
--epochs $epochs \
--third-weight $third_weight \
--use-segment $use_segment
"

decode_opts=\
"\
--ctc-weight $ctc_weight \
--mmi-weight $mmi_weight \
--ngram-weight $ngram_weight \
--mmi-rescore $mmi_rescore \
--beam-size $beam_size \
"

if [ $debug == true ]; then
    ngpu=1
    train_opts="$train_opts \
                --elayers 1 \
                --dlayers 1 \
                --minibatches 200 \
                --report-interval 2"
fi
################# End Paramters ######################


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp
train_dev=dev

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
    # remove space in text
    for x in train dev test; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        # rm data/${x}/text.org # we need this for MMI
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
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/train exp/make_fbank/train ${fbankdir}
    utils/fix_data_dir.sh data/train
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/dev exp/make_fbank/dev ${fbankdir}
    utils/fix_data_dir.sh data/dev
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/test exp/make_fbank/test ${fbankdir}
    utils/fix_data_dir.sh data/test

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    rm -r data/temp1 data/temp2 data/temp3
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
    utils/fix_data_dir.sh data/${train_set}

    # text.org is needed by MMI. Build it in train_sp
    mv data/train_sp/text data/train_sp/text_char
    python3 local/build_sp_text.py data/train/text.org data/train_sp/text
    utils/fix_data_dir.sh data/${train_set}
    mv data/train_sp/text data/train_sp/text.org
    mv data/train_sp/text_char data/train_sp/text

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
                 --text_org data/${train_set}/text.org \
		 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
                     --text_org data/$rtask/text.org \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

lang=data/lang_${ctc_type}_${mmi_unit}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: prepare k2 lang and lm"
    # 1. in espnet oov is <unk> but in k2 it is <UNK>
    #    this would not lead to bug: considered as OOV 
    # 2. <space> is not considered in mandarin but should be considered in English
 
    if [ $mmi_unit == "phone" ]; then
      local/k2_aishell_prepare_dict.sh $data/resource_aishell data/local/dict_nosp

      local/k2_prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
      "<UNK>" data/local/lang_tmp_nosp $lang || exit 1

      local/aishell_train_lms.sh
      gunzip -c data/local/lm/3gram-mincount/lm_unpruned.gz >data/local/lm/lm_tgmed.arpa

      python3 -m kaldilm \
        --read-symbol-table="${lang}/words.txt" \
        --disambig-symbol='#0' \
        --max-order=1 \
        data/local/lm/lm_tgmed.arpa > ${lang}/G_uni.fst.txt

      python3 -m kaldilm \
        --read-symbol-table="${lang}/words.txt" \
        --disambig-symbol='#0' \
        --max-order=3 \
        data/local/lm/lm_tgmed.arpa > ${lang}/G.fst.txt
    
    else
      # We do not need to build ngram LM. Since this MMI will not be used in decoding
      # To train the G.fst, we should split transcription into char level
      local/k2_aishell_prepare_dict_char.sh $dict data/local/dict_nosp
      local/k2_prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
        "<UNK>" data/local/lang_tmp_nosp $lang || exit 1
    fi

fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

ngramexpname=train_ngram
ngramexpdir=exp/${ngramexpname}
if [ -z ${ngramtag} ]; then
    ngramtag=${n_gram}
fi
mkdir -p ${ngramexpdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " \
        > ${lmdatadir}/valid.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
    
    lmplz --discount_fallback -o ${n_gram} <${lmdatadir}/train.txt > ${ngramexpdir}/${n_gram}gram.arpa
    build_binary -s ${ngramexpdir}/${n_gram}gram.arpa ${ngramexpdir}/${n_gram}gram.bin
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Network Training"
    echo "Using override opts: $train_opts"
    echo $train_opts >> ${expdir}/train_opts
  
    if [ ! -d ${feat_tr_dir}/split${ngpu}utt ]; then
        splitjson.py ${feat_tr_dir}/data.json -p $ngpu
    fi

    MASTER_PORT=22277
    NCCL_DEBUG=TRACE python3 -m torch.distributed.launch \
        --nproc_per_node $ngpu --master_port $MASTER_PORT \
        $PWD/espnet/bin/asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu 1 \
        --backend ${backend} \
        --outdir ${expdir}/results_RANK \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/split${ngpu}utt/data.RANK.json \
        --valid-json ${feat_dt_dir}/data.json \
        --ctc_type $ctc_type \
        --lang $lang \
        --opt "noam_sgd" \
        --n-iter-processes 8 \
        --world-size $ngpu \
        $train_opts > ${expdir}/global_record.txt 2>&1
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 6: Decoding"
    nj=$decode_nj
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        recog_model=model.last${idx_average}.avg.best
        average_checkpoints.py --backend ${backend} \
        		       --snapshots ${expdir}/results_0/snapshot.ep.* \
        		       --out ${expdir}/results_0/${recog_model} \
        		       --num ${idx_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_ctc${ctc_weight}_mmi${mmi_weight}_ngram${ngram_weight}_${idx_average}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        echo $decode_opts
        ${decode_cmd} JOB=1:$nj ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results_0/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --ngram-model ${ngramexpdir}/${n_gram}gram.bin \
            --api v2 \
            --local-rank JOB \
            --mmi-type "frame" \
            $decode_opts

        score_sclite.sh ${expdir}/${decode_dir} ${dict} > ${expdir}/${decode_dir}/decode_result.txt
        bash local/mmi_rescore.sh ${expdir}/${decode_dir} $dict
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding with FST"
    nj=1 # 1gpu is enough as it is very fast
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        recog_model=model.${idx_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${idx_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do

        decode_dir=decode_${rtask}_fst_decoding
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### decoding using multiple GPU
        ngpu=1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode_${idx_average}.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 4 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --k2-decode true \
            --local-rank JOB

        tail ${expdir}/${decode_dir}/log/decode_${idx_average}.1.log
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

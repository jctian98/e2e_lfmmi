#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
use_valbest_average=false     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/mnt/ceph_asr_ts/tomasyu/jinchuan/data/librispeech/

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
ngpu=8
tag="8v100_lasmmictc" # tag for managing experiments.
seed=888

# train config
debug=false
batch_size=4
accum_grad=8
epochs=100
mtlalpha=0.5
third_weight=0.5
ctc_type=k2mmi # k2ctc builtin

# decode config
recog_set="test_clean test_other dev_clean dev_other"
idx_average=91_100
lm_weight=0.5
mmi_weight=0.0
ctc_weight=0.0
search_type="alsd"
mmi_type="lookahead"
beam_size=10

. utils/parse_options.sh || exit 1;

train_opts=\
"\
--seed $seed \
--accum-grad $accum_grad \
--batch-size $batch_size \
--epochs $epochs \
--mtlalpha $mtlalpha \
--third-weight $third_weight \
--ctc_type $ctc_type \
"

decode_opts=\
"\
--search-type $search_type \
--lm-weight $lm_weight \
--mmi-weight $mmi_weight \
--mmi-type $mmi_type \
--beam-size $beam_size \
--ctc-weight $ctc_weight \
"

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="9.135.217.29"
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_sp=train_sp
train_dev=dev

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

lang=data/lang_phone
dict=data/lang_char/train_960_unigram5000_units.txt 
feat_tr_dir=${dumpdir}/${train_sp}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Training"

    echo $train_opts >> $expdir/train_opts
    echo $train_opts
    MASTER_PORT=22277
    NCCL_DEBUG=TRACE python3 -m torch.distributed.launch \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $MASTER_PORT \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        $PWD/espnet/bin/asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu 1 \
        --backend ${backend} \
        --outdir ${expdir}/results_RANK \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --world-size $ngpu \
        --opt "noam_sgd" \
        --n-iter-processes 8 \
        --use-segment true \
        --lang $lang \
        --node-rank ${INDEX} \
        --node-size ${HOST_GPU_NUM} \
        --train-json ${feat_tr_dir}/split${ngpu}utt/data_unigram5000.RANK.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
        $train_opts > ${expdir}/global_record.${INDEX}.txt 2>&1 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${idx_average}.avg.best
            opt="--log ${expdir}/results_0/log"
        else
            recog_model=model.last${idx_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results_0/snapshot.ep.* \
            --out ${expdir}/results_0/${recog_model} \
            --num ${idx_average} \

        # download from official repo
        lang=exp/train_rnnlm/rnnlm.model.best
        lang_config=exp/train_rnnlm/model.json
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    
        decode_dir=decode_${rtask}_lm${lm_weight}_${search_type}_mmi${mmi_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        # nj=`wc -l ${feat_recog_dir}/feats.scp | awk '{print $1}'`
        nj=188
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:$nj ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results_0/${recog_model}  \
            --rnnlm ${lang} --rnnlm-conf $lang_config \
            --local-rank JOB $decode_opts -api v2 

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict} > ${expdir}/${decode_dir}/decode_result.txt

    
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

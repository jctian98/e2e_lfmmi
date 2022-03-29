#!/usr/bin/env bash

# author: tyriontian
# tyriontian@tencent.com

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
debug=false

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train_conformer-rnn_transducer_cs.yaml
decode_config=conf/tuning/transducer/decode_default.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
resume=

# data
train_json=dump/jsons/split8utt/zh_en_cs_withcls.RANK.json
valid_json=dump/dev_cs/deltafalse/data_withcls.json
dict=data/dict_cs/dict.txt
nlsyms=data/dict_cs/nlsyms.txt
nbpe=5000
bpemodel=data/dict_cs/bpe.model

### Configurable parameters ###
tag=debug
ngpu=8

# Train config
seed=888
batch_size=16
accum_grad=4
epochs=50
aux_ctc_weight=1.0
cs_lang_weight=0.0
cs_share_encoder=true
cs_share_encoder_layers=9
cs_is_pretrain=false
cs_use_adversial_examples=true
cs_is_ctc_decoder=true
cs_use_mask_predictor=false
enc_block_repeat=3
pretrain_model=

# Decode config 
decode_tag="default_decode"
idx_average=91_100
decode_feature=combine 
search_type="ctc_greedy" # ctc_greedy | ctc_beam | alsd 
beam_size=10
ngram_model=data/ngram/train_cs_5gram.bin
ngram_weight=0.0
rnnlm=exp/train_nnlm_combine/rnnlm.model.best
rnnlm_conf=exp/train_nnlm_combine/model.json
lm_weight=0.0
word_ngram=data/word_ngram/train_combine/
word_ngram_weight=0.0
eng_vocab=None
recog_set="test_zh test_en test_cs"

. utils/parse_options.sh || exit 1;

if [ $debug == true ]; then
    export HOST_GPU_NUM=1
    export HOST_NUM=1
    export NODE_NUM=1
    export INDEX=0
    export CHIEF_IP="9.135.217.29"
fi

train_opts=\
"\
--seed $seed \
--batch-size $batch_size \
--accum-grad $accum_grad \
--epochs $epochs \
--aux-ctc-weight $aux_ctc_weight \
--cs-lang-weight $cs_lang_weight \
--cs-share-encoder $cs_share_encoder \
--cs-share-encoder-layers $cs_share_encoder_layers \
--cs-use-adversial-examples $cs_use_adversial_examples \
--cs-is-pretrain $cs_is_pretrain \
--cs-is-ctc-decoder $cs_is_ctc_decoder \
--cs-use-mask-predictor $cs_use_mask_predictor \
--enc-block-repeat $enc_block_repeat \
"

decode_opts=\
"\
--beam-size $beam_size \
--cs-nt-decode-feature $decode_feature \
--search-type $search_type \
--ngram-model $ngram_model \
--ngram-weight $ngram_weight \
--rnnlm $rnnlm \
--rnnlm-conf $rnnlm_conf \
--lm-weight $lm_weight \
--word-ngram-lower-char false \
--word-ngram $word_ngram \
--word-ngram-weight $word_ngram_weight \
--eng-vocab $eng_vocab \
"

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

expname=${tag}
expdir=exp/${expname}
mkdir -p ${expdir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Pre-Training: CTC bilingual system"
    # keep the model file 
    cp espnet/nets/pytorch_backend/e2e_asr_transducer_cs.py $expdir 
    MASTER_PORT=22275
    NCCL_DEBUG=TRACE python3 -m torch.distributed.launch \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $MASTER_PORT \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        ${MAIN_ROOT}/bin/asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu 1 \
        --backend ${backend} \
        --outdir ${expdir}/pretrain_RANK \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json $train_json \
        --valid-json $valid_json \
        --n-iter-processes 8 \
        --world-size $ngpu \
        --node-rank ${INDEX} \
        --node-size ${HOST_GPU_NUM} \
        --num-save-attention 0 \
        --cs-is-pretrain true $train_opts \
        > ${expdir}/pretrain.${INDEX}.txt 2>&1
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 1: Network Fine-tuning: RNNT bilingual system"
    MASTER_PORT=22275
    NCCL_DEBUG=TRACE python3 -m torch.distributed.launch \
        --nproc_per_node ${HOST_GPU_NUM} --master_port $MASTER_PORT \
        --nnodes=${HOST_NUM} --node_rank=${INDEX} --master_addr=${CHIEF_IP} \
        ${MAIN_ROOT}/bin/asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu 1 \
        --backend ${backend} \
        --outdir ${expdir}/finetuning_RANK \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --train-json $train_json \
        --valid-json $valid_json \
        --n-iter-processes 8 \
        --world-size $ngpu \
        --node-rank ${INDEX} \
        --node-size ${HOST_GPU_NUM} \
        --num-save-attention 0 \
        --resume $pretrain_model \
        --load-trainer-and-opt false \
        $train_opts > ${expdir}/finetuning.${INDEX}.txt 2>&1

       
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    nj=5
    recog_model=model.last${idx_average}.avg.best
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then \
        if [ ! -f ${expdir}/pretrain_0/${recog_model} ]; then
            echo "conduct model average"
            average_checkpoints.py --backend ${backend} \
                                   --snapshots ${expdir}/pretrain_0/snapshot.ep.* \
                                   --out ${expdir}/pretrain_0/${recog_model} \
                                   --num ${idx_average} 
        fi
    fi
    decode_parent_dir=${decode_tag}
    for rtask in ${recog_set}; do
        decode_dir=$decode_parent_dir/${rtask}_${decode_feature}_${search_type}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} --max-job 100 ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/pretrain_0/${recog_model} \
            $decode_opts

        if [ $rtask == "test_cs" ]; then 
            mer_opt="--mer true"
        else
            mer_opt=""
        fi
        
        score_sclite.sh --bpe $nbpe --bpemodel ${bpemodel} --wer true $mer_opt --nlsyms data/dict_cs/nlsyms.txt \
          ${expdir}/${decode_dir} ${dict} > ${expdir}/${decode_dir}/decode_result.txt
    done
    echo "Finished"
fi


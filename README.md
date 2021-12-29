# End-to-end speech secognition toolkit
This is an E2E ASR toolkit modified from Espnet1 (version 0.9.9).  
This is the official implementation of paper [**Consistent Training and Decoding For End-to-end Speech Recognition Using Lattice-free MMI**](https://arxiv.org/abs/2112.02498)  
This is also the official implementation of paper **Improving Mandarin End-to-End Speech Recognition with Word N-gram Language Model** (will be released very soon)  
We achieve state-of-the-art results on two of the most popular results in Aishell-1 and AIshell-2 Mandarin datasets.  

Please feel free to change / modify the code as you like. :)


### Update
- 2021/12/29: Release the first version, which contains all MMI-related features, including MMI training criteria, MMI Prefix Score (for attention-based encoder-decoder, AED) and MMI Alignment Score (For neural transducer, NT).


### Environment:
The main dependencies of this code can be divided into three part: `kaldi`, `espnet` and `k2`.  
1. `kaldi` is mainly used for feature extraction. To install kaldi, please follow the instructions [**here**](https://github.com/kaldi-asr/kaldi).
2. `Espnet` is a open-source end-to-end speech recognition toolkit. please follow the instructions [**here**](https://github.com/espnet/espnet) to install its environment.
    2.1. Pytorch, cudatoolkit, along with many other dependencies will be install automatically during this process.
    2.2. If you are going to use NT models, you are recommend to install a RNN-T warpper. Please run `${ESPNET_ROOT}/tools/installer/install_warp-transducer.sh`  
    2.3. Once you have installed the espnet envrionment successfully, please run `pip uninstall espnet` to remove the `espnet` library. So our code will be used.  
    2.4. Also link the `kaldi` in `${ESPNET_ROOT}`: `ln -s ${KALDI-ROOT} ${ESPNET_ROOT}`
3. `k2` is a python-based FST library. Please follow the instructions [**here**](https://github.com/k2-fsa/k2) to install it. GPU version is required.  
    3.1. We are going to release a word-level N-gram LM scorer very soon. To use this, please also install [**kaldilm**](https://github.com/k2-fsa/k2)
4. There might be some dependency conflicts during building the environment. We report ours below as a reference:  
    4.1 OS: CentOS 7; GCC 7.3.1; Python 3.8.10; CUDA 10.1; Pytorch 1.7.1; k2-fsa 1.2 (very old now)  
    4.2 Other python libraries are in [**requirement.txt**](https://github.com/jctian98/e2e_lfmmi/blob/master/requirement.txt) (It is not recommend to use this file to build the environment directly).

### Results
Currently we have released examples on Aishell-1 and Aishell-2 datasets.  

With MMI training & decoding methods and the word-level N-gram LM. We achieve results on Aishell-1 and Aishell-2 as below. All results are in CER%

|  Test set   | dev | test | ios | android | mic |  
|  :----      | :-: | :--: | :-: | :-----: | :-: |
| AED         | 4.08| 4.45 | 5.26| 6.22    | 5.92|
| NT          | 3.86| 4.18 | 5.06| 6.08    | 5.98|

(example on Librispeech is not fully prepared)

### Get Start
Take Aishell-1 as an example. Working process for other examples are very similar
```
cd ${ESPNET_ROOT}/egs/aishell1
source path.sh
bash prepare.sh # prepare the data

# split the json file of training data for each GPU. (we use 8GPUs)
python3 espnet_utils/splitjson.py -p <ngpu> dump/train_sp/deltafalse/data.json

# these two scripts run independently. 
# stage 1 is for training; stage 2 is for decoding
bash nt.sh      # to train the nueal transducer model
bash aed.sh     # or to train the attention-based encoder-decoder model
```
Several Hint:
1. Please change the paths in `path.sh` accordingly before you start
2. Please change the `data` to config your data path in `prepare.sh`
3. Our code runs in DDP style. Before you start, you need to set them manually. We assume Pytorch distributed API works well on your machine.  
```
export HOST_GPU_NUM=x       # number of GPUs on each host
export HOST_NUM=x           # number of hosts
export NODE_NUM=x           # number of GPUs in total (on all hosts)
export INDEX=x              # index of this host
export CHIEF_IP=xx.xx.xx.xx # IP of the master host
```
4. Multiple choices are available during decoding.  
```
bash aed.sh --stage 2 --mmi-weight 0.2        # activate MMI scorer
bash nt.sh  --stage 2 --mmi-weight 0.2
bash aed.sh --stage 2 --word-ngram-weight 0.4 # activate word-level N-gram LM scorer (available soon)
bash nt.sh  --stage 2 --word-ngram-weight 0.4
bash aed.sh --stage 2 --ngram-weight 0.4      # activate character-level N-gram LM scorer
bash nt.sh  --stage 2 --ngram-weight 0.4
bash aed.sh --stage 2 --lm-weight 0.4         # activate NNLM scorer
bash nt.sh  --stage 2 --lm-weight 0.4
```

### Authorship
Jinchuan Tian;  tianjinchuan@stu.pku.edu.cn or tyriontian@tencent.com  
Jianwei Yu; tomasyu@tencent.com (supervisor)  
Chao Weng; cweng@tencent.com (supervisor)   

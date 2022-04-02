# End-to-end speech secognition toolkit
This is an E2E ASR toolkit modified from Espnet1 (version 0.9.9).  
If this repositry can help you, we will be appreciate if you can star it and cite our papers.

This is the official implementation following papers:  
[**Consistent Training and Decoding For End-to-end Speech Recognition Using Lattice-free MMI**](https://arxiv.org/abs/2112.02498) (Accepted by ICASSP 2022)  
[**Improving Mandarin End-to-End Speech Recognition with Word N-gram Language Model**](https://ieeexplore.ieee.org/document/9721084) (Accepted by SPL)  
[**Integrate Lattice-Free MMI into End-to-End Speech Recognition**](https://arxiv.org/abs/2203.15614) (Submitted to TASLP) 


We achieve state-of-the-art results on two of the most popular results in Aishell-1 and AIshell-2 Mandarin datasets.  
Please feel free to change / modify the code as you like. :)
### Update
- 2021/12/29: Release the first version, which contains all MMI-related features, including MMI training criteria, MMI Prefix Score (for attention-based encoder-decoder, AED) and MMI Alignment Score (For neural transducer, NT).
- 2022/1/6: Release the word-level N-gram LM scorer.
- 2022/1/12: We update the instructions to build the environment. We also release the trained NT model for Aishell-1 for quick performance check. We update the guildline to run our code.
- 2022/3/29 We release a new CTC / RNN-T recipe for code-switch problem based on ASRU 2019 Mandarin-English code-switch dataset (see egs/asrucs); Results on Aishell-1 and Aishell-2 are also updated.

### Environment:
The main dependencies of this code can be divided into three part: `kaldi`, `espnet` and `k2`  
Please follow the instructions in [build_env.sh](https://github.com/jctian98/e2e_lfmmi/blob/master/env/build_env.sh) to build the environment.  
Note the script cannot run automatically and you need to run it line-by-line.
### Results
Currently we have released examples on Aishell-1 and Aishell-2 datasets.  
With MMI training & decoding methods and the word-level N-gram LM. We achieve results on Aishell-1 and Aishell-2 as below. All results are in CER%  
The model file of Aishell-1 NT system is [here](https://drive.google.com/file/d/1VE2YtLb70UpQkeGWE8WhHJl7sSwNa_zG/view?usp=sharing) for quick performance check.

|  Test set                      | Aishell-1-dev | Aishell-1-test | Aishell-2-ios | Aishell-2-android | Aishell-2-mic |  
|  :----                         | :-: | :--: | :-: | :-----: | :-: |
| AED                            | 4.60| 5.07 | 5.72| 6.60    | 6.58| 
| AED + MMI + Word Ngram         | 4.08| 4.45 | 5.15| 5.92    | 5.77|
| NT                             | 4.41| 4.82 | 5.81| 6.52    | 6.52|
| NT + MMI + Word Ngram          | 3.79| 4.10 | 5.02| 5.85    | 5.66|
 
### Get Start
Take Aishell-1 as an example. Working process for other examples are very similar.  
step 1: clone the code and link kaldi
```
conda activate lfmmi
git clone https://github.com/jctian98/e2e_lfmmi E2E-ASR-Framework # clone and RENAME
cd E2E-ASR-Framework
ln -s <path-to-kaldi> kaldi                                       # link kaldi
```
step 2: prepare data, lexicon and LMs. Before you run, please set the datadir in `prepare.sh`
```
cd egs/aishell1
bash prepare.sh 
```
step 3: model training. You should split the data before start the training.  
You can skip this step and download our trained model [here](https://drive.google.com/file/d/1VE2YtLb70UpQkeGWE8WhHJl7sSwNa_zG/view?usp=sharing)
```
python3 espnet_utils/splitjson.py -p <ngpu> dump/train_sp/deltafalse/data.json
bash nt.sh --stop_stage 1
```
step 4: decode 
```
bash nt.sh --stage 2 --mmi-weight 0.2 --word-ngram-weight 0.4
```
Several Hint:
1. Please change the paths in `path.sh` accordingly before you start
2. Please change the `data` to config your data path in `prepare.sh`
3. Our code runs in DDP style and requires some global variables. Before you start, you need to set them manually. We assume Pytorch distributed API works well on your machine.  
```
export HOST_GPU_NUM=x       # number of GPUs on each host
export HOST_NUM=x           # number of hosts
export NODE_NUM=x           # number of GPUs in total (on all hosts)
export INDEX=x              # index of this host
export CHIEF_IP=xx.xx.xx.xx # IP of the master host
```
4. You may encounter some problem about `k2`. Try to delete `data/lang_phone/Linv.pt` (in training) and `data/word_3gram/G.pt`(in decoding) and re-generate them again. 
5. Multiple choices are available during decoding (we take `nt.sh` as an example, but the usage of `aed.sh` is the same).  
   To use the MMI-related scorers, you need train the model with MMI auxiliary criterion;  
   
  To use MMI Prefix Score (in AED) or MMI Alignment score (in NT):
  ```
  bash nt.sh --stage 2 --mmi-weight 0.2
  ```
  To use any external LM, you need to train them in advance (as implemented in `prepare.sh`)  
  
  To use word-level N-gram LM:
  ```
  bash nt.sh --stage 2 --word-ngram-weight 0.4
  ```
  To use character-level N-gram LM:
  ```
  bash nt.sh --stage 2 --ngram-weight 1.0
  ```
  To use neural network LM:
  ```
  bash nt.sh --stage 2 --lm-weight 1.0
  ```
### Reference
kaldi: https://github.com/kaldi-asr/kaldi  
Espent: https://github.com/espnet/espnet  
k2-fsa: https://github.com/k2-fsa/k2  
### Citations
```
@article{tian2021consistent,  
  title={Consistent Training and Decoding For End-to-end Speech Recognition Using Lattice-free MMI},  
  author={Tian, Jinchuan and Yu, Jianwei and Weng, Chao and Zhang, Shi-Xiong and Su, Dan and Yu, Dong and Zou, Yuexian},  
  journal={arXiv preprint arXiv:2112.02498},  
  year={2021}  
}  

@ARTICLE{9721084,
  author={Tian, Jinchuan and Yu, Jianwei and Weng, Chao and Zou, Yuexian and Yu, Dong},
  journal={IEEE Signal Processing Letters}, 
  title={Improving Mandarin End-to-End Speech Recognition with Word N-gram Language Model}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LSP.2022.3154241}}
  
@article{tian2022integrate,
  title={Integrate Lattice-Free MMI into End-to-End Speech Recognition},
  author={Tian, Jinchuan and Yu, Jianwei and Weng, Chao and Zou, Yuexian and Yu, Dong},
  journal={arXiv preprint arXiv:2203.15614},
  year={2022}
}
```
### Authorship
Jinchuan Tian;  tianjinchuan@stu.pku.edu.cn or tyriontian@tencent.com  
Jianwei Yu; tomasyu@tencent.com (supervisor)  
Chao Weng; cweng@tencent.com  
Yuexian Zou; zouyx@pku.edu.cn

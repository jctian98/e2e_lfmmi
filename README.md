# End-to-end speech secognition toolkit
This is an E2E ASR toolkit modified from Espnet1 (version 0.9.9).   

### docker images / Environment:
We run all code in a docker image. However, this image cannot be released here.
You can build the image by yourself, or just use conda environment. 
Basically, we need a standard Espnet envrionment (see https://github.com/espnet/espnet) with RNN-T warpper extension. Then you install the k2 library (see https://github.com/k2-fsa/k2). Kaldi is used for feature extraction (see https://github.com/kaldi-asr/kaldi). There might be some other dependencies problems but they will be very easy to solve by pip.

### Examples
All examples are in directory `egs/`. Currently the egs of Aishell-1, Aishell-2 and Librispeech are released. More egs will be released in the future. 
To run an eg, you firstly run `prepare.sh` to prepare the data. Then run `aed.sh` to run hybrid CTC/Attention system, or run `nt.sh` to run the RNN-T system.

### Paper:
Consistent Training and Decoding For End-to-end Speech Recognition Using Lattice-free MMI 
Link: https://arxiv.org/abs/2112.02498; paper submitted to ICASSP 2022

### Authorship
Jinchuan Tian;  tianjinchuan@stu.pku.edu.cn or tyriontian@tencent.com;
Jianwei Yu; tomasyu@tencent.com  
Chao Weng; cweng@tencent.com (supervisor)  

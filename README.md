# End-to-end speech secognition toolkit
This is an E2E ASR toolkit modified from Espnet1 (version 0.9.9).   

### docker images:
`mirrors.tencent.com/tyriontian/pytorch1.8.1_cuda10.1_k2_espnet:rnnt`  
All dependecy of this repositry is ensured in this docker image  
Each time you use this image, remember to use `source ~/.bashrc` to activate the virtual environment  
The image would be updated sometimes. Please try to pull the latest version.  

There is a plan to shift the image to a DDP accelerated version (ongoing)  

### Examples
All examples are in directory `egs/`. You can run these examples by `bash run.sh`  
(1) For every dataset we provide the recipe for both neural transducer (RNN-T) and Hybrid CTC/Attention (LAS-CTC). See the configs in the header of `run.sh`  
(2) You may need to change the root of data path to run the scripts. 
(3) Standard Kaldi is also used for feature extraction. Change the `KALDI_ROOT` in `egs/*/path.sh` if you try to extract the features.  

### Highlight Features
(1) This repositry implements LF-MMI in both training and decoding of RNN-T and LASCTC. See our paper *CONSISTENT TRAINING AND DECODING FOR END-TO-END SPEECH RECOGNITION USING LATTICE-FREE MMI*  
(2) We implement a high-performance dataloader for large-scale training. This loader could be transplant to every other ASR codebase. See `interface/dataloader` (ongoing)   
(3) We also provide the model file of our RNN-T implementation. This is for C++ libtorch engineering. (ongoing)  

### Author
Jinchuan Tian; tyriontian@tencent.com  
Jianwei Yu; tomasyu@tencent.com  
Chao Weng; cweng@tencent.com (supervisor)  

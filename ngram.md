# Word-level N-gram scorer for End-to-End speech recognition 

### Introduction
This repositry provides a word-level N-gram LM scorer that could be used in end-to-end speech recognition. Our scorer is compatible with both Hybrid CTC-Attention and Neural Transducers. For any input texts, it provides the posterior log p(w_u|W_1^{u-1}) for online joint decoding. 

To the best of our knowledge, we achieve state-of-the-art performance on Aishell-1 and Aishell-2.

Paper is under writing. 
Full training and decoding pipeline would be released later in https://github.com/jctian98/e2e_lfmmi

### Dependency:
pytorch: https://pytorch.org/  
k2: https://github.com/k2-fsa/k2

### Performance
System performance w/o word-level N-gram LM model. Character Error Rate (CER%) is reported.  
No any external resources used. 
The recommended weight of the LM is 0.3; order of the N-gram LM is 3. 

| Model   | + word N-gram |Aishell-1 dev | Aishell-1 test | Aishell-2 ios | Aishell-2 android | Aishell-2 mic |
| -------------  |:-------------:|:---:|:---:|:---:|:---:|:---:|
| NT + MMI + CTC |False          | 4.2 | 4.5 | 5.4 | 6.6 | 6.5 |
| NT + MMI + CTC |True           | 3.8 | 4.2 | 5.1 | 6.1 | 6.0 |
| HCA            |False          | 4.6 | 5.0 | 5.9 | 7.0 | 6.8 |
| HCA            |True           | 4.2 | 4.6 | 5.3 | 6.2 | 6.0 |

### Acknowledgement
Jinchuan Tian (tianjinchuan@stu.pku.edu.cn; tyriontian@tencent.com)       
Jianwei Yu (tomasyu@tencent.com)    
Chao Weng (cweng@tencent.com)    

This work is performed when Jinchuan Tian was an intern in Tencent AI LAB

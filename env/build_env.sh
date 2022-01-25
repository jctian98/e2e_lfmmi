# Author: Jinchuan Tian; tianjinchuan@stu.pku.edu.cn
# Build the environment for ASR repositry https://github.com/jctian98/e2e_lfmmi

# Here is only an example, you may need to revise this script to suit your machine
# This script can hardly run automatically. You may need to run it line-by-line

# Our system:
# Centos 7; GCC 7.3.1
# Python 3.8 + CUDA 10.1 + Pytorch 1.7.1, 

rootdir=/home/tian/tools/opensource
stage=$1
nj=48

cd $rootdir
if [ ${stage} -le 1 ]; then
  echo "Install GCC 7.3.1 and system dependency. You need root account for this"
  yum install -y cmake sox libsndfile ffmpeg flac

  yum install -y centos-release-scl
  yum install -y devtoolset-7
  scl enable devtoolset-7 bash
  # After this, run 'gcc -v' to ensure the GCC version is correct
fi

if [ ${stage} -le 2 ]; then
  echo "Install Kaldi and its auxiliary tools"
  git clone https://github.com/kaldi-asr/kaldi.git
  cd kaldi/tools/
  bash extras/check_dependencies.sh # make sure it's ok

  make -j $nj
  cd ../src/
  ./configure --shared
  make depend -j $nj
  make -j $nj

  # Additionally, you need kaldi_lm to train word-level N-gram LM
  cd ../tools
  bash extras/install_kaldi_lm.sh
fi

if [ ${stage} -le 3 ]; then
  echo "Install Espnet environment"
  # git clone https://github.com/espnet/espnet
  cd $rootdir/espnet/tools
  ln -s $rootdir/kaldi .

  # Build espnet environment. You may choose other versions
  CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
  ./setup_anaconda.sh ${CONDA_TOOLS_DIR} lfmmi 3.8
  make TH_VERSION=1.7.1 CUDA_VERSION=10.1 

  # NT warpper is required if you will run NT examples
  # Installing this warpper is difficult. This issue might be
  # helpful: https://github.com/HawkAaron/warp-transducer/pull/90
  # Also in our installing process, we find the pytorch test
  # cannot pass as the gradients mismatch the desired value
  installers/install_warp-transducer.sh 

  # to use our code rather than standard espnet code
  pip3 uninstall espnet
fi

if [ ${stage} -le 4 ]; then
  echo "Install k2 library"
  conda install -c k2-fsa -c pytorch -c conda-forge k2 python=3.8 cudatoolkit=10.1 pytorch=1.7.1

fi

if [ ${stage} -le 5 ]; then
  echo "Install other python libraries"
  pip3 install kaldilm chainer==6.0.0 kaldialign graphviz lhotse numpy==1.20 
fi


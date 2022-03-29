# this is necessary since docker images would not run .bashrc if the command line is not "bash"
source ~/.bashrc # to include libfst.so

MAIN_ROOT=$PWD/../../
KALDI_ROOT=../../kaldi/ # Kaldi is local and is not available on jizhi task

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=$PWD/espnet_utils:$MAIN_ROOT/bin:$MAIN_ROOT:$PATH

export OMP_NUM_THREADS=1
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PATH=${KALDI_ROOT}/tools/sph2pipe:${KALDI_ROOT}/tools/kaldi_lm:${KALDI_ROOT}/tools/sctk/bin:${KALDI_ROOT}/tools/kenlm/build/bin:${KALDI_ROOT}/tools/srilm/bin/i686-m64/:${KALDI_ROOT}/tools/srilm/lm/bin/i686-m64/:$PATH
export PYTHONPATH=$MAIN_ROOT:$MAIN_ROOT/bin/:$MAIN_ROOT/../:$PWD:$PYTHONPATH # so espnet could be find like a library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:${KALDI_ROOT}/tools/openfst/lib/
# nvidia-smi -c 3

#!/usr/bin/env bash

# Copyright 2014 Vassil Panayotov
# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Prepares the dictionary

stage=3

if [ $# -ne 2 ]; then
  echo "Usage: $0 <lm-dir> <dst-dir>"
  echo "e.g.: /export/a15/vpanayotov/data/lm data/local/dict"
  exit 1
fi

lm_dir=$1
dst_dir=$2

vocab=$lm_dir/librispeech-vocab.txt
[ ! -f $vocab ] && echo "$0: vocabulary file not found at $vocab" && exit 1;

# this file is either a copy of the lexicon we download from openslr.org/11 or is
# created by the G2P steps below
lexicon_raw_nosil=$dst_dir/lexicon_raw_nosil.txt

mkdir -p $dst_dir || exit 1;

# The copy operation below is necessary, if we skip the g2p stages(e.g. using --stage 3)
if [[ ! -s "$lexicon_raw_nosil" ]]; then
  cp $lm_dir/librispeech-lexicon.txt $lexicon_raw_nosil || exit 1
fi

if [ $stage -le 3 ]; then
  silence_phones=$dst_dir/silence_phones.txt
  optional_silence=$dst_dir/optional_silence.txt
  nonsil_phones=$dst_dir/nonsilence_phones.txt

  echo "Preparing phone lists"
  (echo SIL; echo SPN;) > $silence_phones
  echo SIL > $optional_silence
  # nonsilence phones; on each line is a list of phones that correspond
  # really to the same base phone.
  awk '{for (i=2; i<=NF; ++i) { print $i; gsub(/[0-9]/, "", $i); print $i}}' $lexicon_raw_nosil |\
    sort -u |\
    perl -e 'while(<>){
      chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_";
      $phones_of{$1} .= "$_ "; }
      foreach $list (values %phones_of) {print $list . "\n"; } ' | sort \
      > $nonsil_phones || exit 1;
fi

if [ $stage -le 4 ]; then
  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |\
  cat - $lexicon_raw_nosil | sort | uniq >$dst_dir/lexicon.txt
  echo "Lexicon text file saved as: $dst_dir/lexicon.txt"
fi

exit 0

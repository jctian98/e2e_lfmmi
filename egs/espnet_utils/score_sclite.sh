#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

[ -f ./path.sh ] && . ./path.sh

nlsyms=""
wer=false
mer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""
num_spkrs=1
help_message="Usage: $0 <data-dir> <dict>"
sppd3=false

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1
dic=$2

concatjson.py ${dir}/data.*.json > ${dir}/data.json

if [ $num_spkrs -eq 1 ]; then
  json2trn.py ${dir}/data.json ${dic} --num-spkrs ${num_spkrs} --refs ${dir}/ref.trn --hyps ${dir}/hyp.trn

  if ${remove_blank}; then
      sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
  fi
  if [ -n "${nlsyms}" ]; then
      cp ${dir}/ref.trn ${dir}/ref.trn.org
      cp ${dir}/hyp.trn ${dir}/hyp.trn.org
      filt.py -v ${nlsyms} ${dir}/ref.trn.org > ${dir}/ref.trn
      filt.py -v ${nlsyms} ${dir}/hyp.trn.org > ${dir}/hyp.trn
  fi
  if [ -n "${filter}" ]; then
      sed -i.bak3 -f ${filter} ${dir}/hyp.trn
      sed -i.bak3 -f ${filter} ${dir}/ref.trn
  fi

  if [ $sppd3 = true ]; then
      cp ${dir}/hyp.trn ${dir}/hyp.trn.org 
      python3 espnet_utils/filter_trn.py $dir/hyp.trn.org > ${dir}/hyp.trn
  fi

  sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt
  
  echo "write a CER (or TER) result in ${dir}/result.txt"
  grep -e Avg -e SPKR -m 2 ${dir}/result.txt
  python3 espnet_utils/double_precious_cer.py ${dir}/result.txt

  if ${wer}; then
      if [ -n "$bpe" ]; then
  	    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
  	    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
      else
  	    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
  	    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
      fi
      sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt

      echo "write a WER result in ${dir}/result.wrd.txt"
      grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
      python3 espnet_utils/double_precious_cer.py ${dir}/result.wrd.txt

      if ${mer}; then
         python3 espnet_utils/prepare_mer.py ${dir}/ref.wrd.trn ${dir}/ref.wrd.trn.chn ${dir}/ref.wrd.trn.eng
         python3 espnet_utils/prepare_mer.py ${dir}/hyp.wrd.trn ${dir}/hyp.wrd.trn.chn ${dir}/hyp.wrd.trn.eng 
         sclite -r ${dir}/ref.wrd.trn.chn trn -h ${dir}/hyp.wrd.trn.chn trn -i rm -o all stdout > ${dir}/result.wrd.chn.txt
         sclite -r ${dir}/ref.wrd.trn.eng trn -h ${dir}/hyp.wrd.trn.eng trn -i rm -o all stdout > ${dir}/result.wrd.eng.txt
         
         echo "write a Mandarin CER result of code-switch data in ${dir}/result.wrd.chn.txt"
         grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.chn.txt
         echo "write a English MER result of code-switch data in ${dir}/result.wrd.eng.txt"
         grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.eng.txt
      fi
  fi
elif [ ${num_spkrs} -lt 4 ]; then
  ref_trns=""
  hyp_trns=""
  for i in $(seq ${num_spkrs}); do
      ref_trns=${ref_trns}"${dir}/ref${i}.trn "
      hyp_trns=${hyp_trns}"${dir}/hyp${i}.trn "
  done
  json2trn.py ${dir}/data.json ${dic} --num-spkrs ${num_spkrs} --refs ${ref_trns} --hyps ${hyp_trns}

  for n in $(seq ${num_spkrs}); do
      if ${remove_blank}; then
          sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp${n}.trn
      fi
      if [ -n "${nlsyms}" ]; then
          cp ${dir}/ref${n}.trn ${dir}/ref${n}.trn.org
          cp ${dir}/hyp${n}.trn ${dir}/hyp${n}.trn.org
          filt.py -v ${nlsyms} ${dir}/ref${n}.trn.org > ${dir}/ref${n}.trn
          filt.py -v ${nlsyms} ${dir}/hyp${n}.trn.org > ${dir}/hyp${n}.trn
      fi
      if [ -n "${filter}" ]; then
          sed -i.bak3 -f ${filter} ${dir}/hyp${n}.trn
          sed -i.bak3 -f ${filter} ${dir}/ref${n}.trn
      fi
  done

  results_str=""
  for (( i=0; i<$((num_spkrs * num_spkrs)); i++ )); do
      ind_r=$((i / num_spkrs + 1))
      ind_h=$((i % num_spkrs + 1))
      results_str=${results_str}"${dir}/result_r${ind_r}h${ind_h}.txt "
      sclite -r ${dir}/ref${ind_r}.trn trn -h ${dir}/hyp${ind_h}.trn trn -i rm -o all stdout > ${dir}/result_r${ind_r}h${ind_h}.txt
  done

  echo "write CER (or TER) results in ${dir}/result_r*h*.txt"
  eval_perm_free_error.py --num-spkrs ${num_spkrs} \
      ${results_str} > ${dir}/min_perm_result.json
  sed -n '2,4p' ${dir}/min_perm_result.json

  if ${wer}; then
      for n in $(seq ${num_spkrs}); do
          if [ -n "$bpe" ]; then
              spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref${n}.trn | sed -e "s/▁/ /g" > ${dir}/ref${n}.wrd.trn
              spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp${n}.trn | sed -e "s/▁/ /g" > ${dir}/hyp${n}.wrd.trn
          else
              sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref${n}.trn > ${dir}/ref${n}.wrd.trn
              sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp${n}.trn > ${dir}/hyp${n}.wrd.trn
          fi
      done
      results_str=""
      for (( i=0; i<$((num_spkrs * num_spkrs)); i++ )); do
          ind_r=$((i / num_spkrs + 1))
          ind_h=$((i % num_spkrs + 1))
          results_str=${results_str}"${dir}/result_r${ind_r}h${ind_h}.wrd.txt "
          sclite -r ${dir}/ref${ind_r}.wrd.trn trn -h ${dir}/hyp${ind_h}.wrd.trn trn -i rm -o all stdout > ${dir}/result_r${ind_r}h${ind_h}.wrd.txt
      done

      echo "write WER results in ${dir}/result_r*h*.wrd.txt"
      eval_perm_free_error.py --num-spkrs ${num_spkrs} \
          ${results_str} > ${dir}/min_perm_result.wrd.json
      sed -n '2,4p' ${dir}/min_perm_result.wrd.json
  fi
fi

decode_dir=$1
word_ngram=$2
dict=$3

. utils/parse_options.sh || exit 1;

mkdir -p $decode_dir/rescore

for w in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    (mkdir -p $decode_dir/word_ngram_rescore/$w; subdir=$decode_dir/word_ngram_rescore/$w
    python3 espnet_utils/word_ngram_rescore.py $decode_dir/data.json \
      $word_ngram $w $subdir/data.1.json
    bash espnet_utils/score_sclite.sh $subdir $dict \
      > $subdir/decode_result.txt) & 
done
wait 

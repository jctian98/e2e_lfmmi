decode_dir=$1
dict=$2

mkdir -p $decode_dir/rescore
dir=$decode_dir/rescore

mkdir -p $dir/best
python3 local/max_rescore.py $decode_dir/data.json ${dir}/best/data.1.json $dir/best/best.json
score_sclite.sh  $dir/best ${dict} > ${dir}/best/decode_result.txt

for w in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    mkdir -p $dir/$w
    python3 local/rerank.py $decode_dir/data.json $w ${dir}/${w}/data.1.json
    score_sclite.sh  $dir/$w ${dict} > ${dir}/$w/decode_result.txt
done 

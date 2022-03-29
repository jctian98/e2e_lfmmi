decode_dir=$1
dict=$2

mkdir -p $decode_dir/rescore
dir=$decode_dir/rescore

mkdir -p $dir/best

for w in 0.05 0.1 0.2 0.3; do
    mkdir -p $dir/$w
    (python3 espnet_utils/rerank_mmi.py $decode_dir/data.json $w ${dir}/${w}/data.1.json
    score_sclite.sh  --sppd3 true $dir/$w ${dict} > ${dir}/$w/decode_result.txt) &
done
wait 

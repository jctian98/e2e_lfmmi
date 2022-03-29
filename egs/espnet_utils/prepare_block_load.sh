num_split=32
bpe_model=

. utils/parse_options.sh

kdir=$1 # kaldi dataset directory
ddir=$2 # dump directory
dst=$3 # distination directory
dict=$4

# step 1: sort the scp in dumpdir according to the utt2num_frames
mkdir -p $dst
tmpdir=$dst/tmp; mkdir -p $tmpdir
python3 espnet_utils/sort_scp_by_length.py $ddir/feats.scp $ddir/utt2num_frames \
                                   $tmpdir/feats.scp $tmpdir/utt2num_frames

# step 2: split the feats.scp and utt2num_frames according to  `num_split`
fests_scps=""
for idx in `seq 1 $num_split`; do
   dir=$dst/$idx; mkdir -p $dir
   feats_scps="$feats_scps $dir/feats.scp"
done
python3 espnet_utils/split_scp.py $tmpdir/feats.scp $feats_scps

for idx in `seq 1 $num_split`; do
   dir=$dst/$idx;
   python3 espnet_utils/filter_scp.py $dir/feats.scp \
       $tmpdir/utt2num_frames > $dir/utt2num_frames &
done
wait

# step 3: copy-feats
for idx in `seq 1 $num_split`; do
   dir=$dst/$idx;
   python3 espnet_utils/split_scp_fix_length.py $dir/feats.scp
   nj=`ls $dir/feats.*.scp | wc -l`
   ${decode_cmd} JOB=1:$nj $dir/copy_logs/copy_feat.JOB.log \
       copy-feats --compress=true --compression-method=2 \
           scp:$dir/feats.JOB.scp \
           ark,scp:$dir/feats.JOB.ark,$dir/feats_copy.JOB.scp
   for j in `seq 1 $nj`; do
       cat $dir/feats_copy.${j}.scp 
   done > $dir/feats.scp
done

# step 4: filter the kaldi format data
for idx in `seq 1 $num_split`; do
    dir=$dst/$idx;
    mkdir -p $dir/kaldi_files
    for f in text utt2spk spk2utt text_org; do
        if [ -f  $kdir/$f ]; then
        python3 espnet_utils/filter_scp.py $dir/feats.scp \
            $kdir/$f > $dir/kaldi_files/$f &
        fi
    done
    wait
done

# step 5: make json
for idx in `seq 1 $num_split`; do
    dir=$dst/$idx;
    
    if [ -f $dir/kaldi_files/text_org ]; then
        json_opts="--text_org $dir/kaldi_files/text_org"
    else
        json_opts=""
    fi

    if [ ! -z $bpe_model ]; then
        json_opts="$json_opts --bpecode $bpe_model"
    else
        json_opts="$json_opts"
    fi

    bash espnet_utils/data2json.sh $json_opts --feat $dir/feats.scp \
        $dir/kaldi_files $dict > $dir/data.json &
done
wait

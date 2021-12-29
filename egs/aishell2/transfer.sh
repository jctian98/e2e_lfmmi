src=
dst=

for x in train_sp_pytorch_8v100_rnnt_mmi; do 
    mkdir -p $dst/$x
    mkdir -p $dst/$x/results_0
    (cp $src/$x/results_0/snapshot* $dst/$x/results_0/
    cp $src/$x/results_0/model.json $dst/$x/results_0/) &
done
wait
    

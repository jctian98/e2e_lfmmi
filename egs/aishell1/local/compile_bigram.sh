# Compile char level bigram LM. for MMI training. 
# The bigram should be sparse or 4300+ words would lead to 17M arcs and overflow of GPU memory

lang=$1
train_text=$2
threshold=2

lmplz -o 2 --prune $threshold < $train_text > $lang/P.arpa
python3 -m kaldilm \
        --read-symbol-table="${lang}/words.txt" \
        --disambig-symbol='#0' \
        --max-order=2 \
        $lang/P.arpa > ${lang}/P.fst.txt

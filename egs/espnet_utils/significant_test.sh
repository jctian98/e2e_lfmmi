adir=$1 # reference
bdir=$2 # tested 

for part in trn wrd.trn wrd.trn.chn wrd.trn.eng; do
    if [ -f $adir/ref.$part ] && [ -f $adir/ref.$part ]; then 
        (sclite -F -i wsj -r $adir/ref.$part -h $adir/hyp.$part -o sgml
        sclite -F -i wsj -r $bdir/ref.$part -h $bdir/hyp.$part -o sgml

        cat $adir/hyp.${part}.sgml $bdir/hyp.${part}.sgml | sc_stats -p -t mapsswe -v -u -n $bdir/result.${part}.mapsswe
        ) &
    fi
done
wait 

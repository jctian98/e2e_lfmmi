# Build character-level dict for K2 CTC / MMI 
# The token list would be very large (10k+) if we use aishell lexicon 
# so we use the token list of espnet
[ $# != 2 ] && echo "Usage: $0 <espnet-char-list> <dest-path>" && exit 1

lex=$1
dict=$2

rm -r $dict
mkdir -p $dict

# prepare lexicon
cat $lex | tail -n +2 | awk '{print $1, $1}' > $dict/lexicon.txt
echo "<UNK> spn" >> $dict/lexicon.txt
echo "SIL sil" >> $dict/lexicon.txt
echo "<SPOKEN_NOISE> sil" >> $dict/lexicon.txt

# phones and extra questions
echo sil >$dict/silence_phones.txt
echo sil >$dict/optional_silence.txt
echo sil >$dict/extra_questions.txt
cat $dict/lexicon.txt | cut -d " " -f 2 | grep -v "sil" > $dict/nonsilence_phones.txt

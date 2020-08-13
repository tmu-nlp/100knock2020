SRC=use_file/tok
DST=use_file_bpe
BPEROOT=fairseq/examples/translation/subword-nmt/subword_nmt
BPE_TOKENS=10000
BPE_CODE=$DST/code

echo "learn_bpe.py"
mkdir -p $DST
cat $SRC/kyoto-train.cln.ja $SRC/kyoto-train.cln.en >$SRC/kyoto-train.ja-en
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS <$SRC/kyoto-train.ja-en >$BPE_CODE

for file in $(find $SRC -type f); do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $file >$DST/$(basename $file)
done

set -x

src=ja
tgt=en
prep=use_file_bpe

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $prep/kyoto-train.cln --validpref $prep/kyoto-dev --testpref $prep/kyoto-test \
    --destdir $prep/AT_bin \
    --bpe subword_nmt \
    --workers 20
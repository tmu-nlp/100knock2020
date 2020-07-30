fairseq-preprocess -s en -t de \
    --trainpref train-bpe \
    --validpref valid-bpe \
    --testpref test-bpe \
    --destdir data-en-de-bpe-prep95 \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20

# | [en] Dictionary: 6199 types
# | [en] train-bpe.en: 160239 sents, 3949114 tokens, 0.022% replaced by <unk>
# | [en] Dictionary: 6199 types
# | [en] valid-bpe.en: 7283 sents, 178622 tokens, 0.0252% replaced by <unk>
# | [en] Dictionary: 6199 types
# | [en] test-bpe.en: 6750 sents, 156928 tokens, 0.0287% replaced by <unk>
# | [de] Dictionary: 7863 types
# | [de] train-bpe.de: 160239 sents, 4035591 tokens, 0.0494% replaced by <unk>
# | [de] Dictionary: 7863 types
# | [de] valid-bpe.de: 7283 sents, 182592 tokens, 0.0761% replaced by <unk>
# | [de] Dictionary: 7863 types
# | [de] test-bpe.de: 6750 sents, 161838 tokens, 0.146% replaced by <unk>
# | Wrote preprocessed data to data-en-de-bpe-prep95

fairseq-train data-en-de-bpe-prep95 \
    --fp16 \
    --save-dir save95-lstm-en-de-bpe \
    --max-epoch 10 \
    --lr 1e-3 \
    --optimizer adam --clip-norm 0.1 \
    --dropout 0.2 \
    --max-tokens 4000 \
    --arch lstm

fairseq-interactive --path save95-lstm-en-de-bpe/checkpoint_best.pt data-en-de-bpe-prep95 < test-bpe.en | grep '^H' | cut -f3 > knock95-pred.out

fairseq-score --sys knock95-pred.out --ref test-bpe.en
# Namespace(ignore_case=False, order=4, ref='test-bpe.en', sacrebleu=False, sentence_bleu=False, sys='knock95-pred.out')
# BLEU4 = 4.10, 19.3/4.9/2.4/1.3 (BP=0.984, ratio=0.984, syslen=147819, reflen=150178)

for N in `seq 1 20`; do
    fairseq-interactive --path save95-lstm-en-de-bpe/checkpoint_best.pt --beam $N data-en-de-bpe-prep95 < test-bpe.en | grep '^H' | cut -f3 > knock95.$N.out
done

for N in `seq 1 20` ; do
    fairseq-score --sys knock95.$N.out --ref test-bpe.en > knock95.$N.score
done
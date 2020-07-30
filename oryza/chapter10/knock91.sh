fairseq-preprocess -s en -t de \
    --trainpref train.de-en \
    --validpref valid.de-en \
    --testpref test.de-en \
    --destdir data-en-de-prep91 \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20

# Namespace(align_suffix=None, alignfile=None, bpe=None, cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='data-prep91', empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_format=None, log_interval=1000, lr_scheduler='fixed', memory_efficient_fp16=False, min_loss_scale=0.0001, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, only_source=False, optimizer='nag', padding_factor=8, seed=1, source_lang='de', srcdict=None, target_lang='en', task='translation', tensorboard_logdir='', testpref='test.de-en', tgtdict=None, threshold_loss_scale=None, thresholdsrc=5, thresholdtgt=5, tokenizer=None, trainpref='train.de-en', user_dir=None, validpref='valid.de-en', workers=20)
# | [de] Dictionary: 21031 types
# | [de] train.de-en.de: 153348 sents, 2841735 tokens, 4.34% replaced by <unk>
# | [de] Dictionary: 21031 types
# | [de] valid.de-en.de: 6970 sents, 129064 tokens, 4.84% replaced by <unk>
# | [de] Dictionary: 21031 types
# | [de] test.de-en.de: 6750 sents, 132504 tokens, 4.99% replaced by <unk>
# | [en] Dictionary: 16423 types
# | [en] train.de-en.en: 153348 sents, 2990588 tokens, 1.88% replaced by <unk>
# | [en] Dictionary: 16423 types
# | [en] valid.de-en.en: 6970 sents, 135929 tokens, 2.14% replaced by <unk>
# | [en] Dictionary: 16423 types
# | [en] test.de-en.en: 6750 sents, 137906 tokens, 2.21% replaced by <unk>
# | Wrote preprocessed data to data-prep91

fairseq-train data-en-de-prep91 \
    --fp16 \
    --save-dir save91-lstm-en-de \
    --max-epoch 10 \
    --lr 1e-3 \
    --optimizer adam --clip-norm 0.1 \
    --dropout 0.2 \
    --max-tokens 4000 \
    --arch lstm
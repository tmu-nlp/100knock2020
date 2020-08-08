set -x

src=$1
tgt=$2
prep=use_file

fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $prep/kyoto-train.cln --validpref $prep/kyoto-dev --testpref $prep/kyoto-test \
    --destdir $prep/AT_bin \
    --workers 20

<< result

++ src=ja
++ tgt=en
++ prep=use_file
++ fairseq-preprocess --source-lang ja --target-lang en --trainpref use_file/kyoto-train.cln --validpref use_file/kyoto-dev --testpref use_file/kyoto-test --destdir use_file/AT_bin --workers 20
Namespace(align_suffix=None, alignfile=None, bpe=None, cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='use_file/AT_bin', empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_format=None, log_interval=1000, lr_scheduler='fixed', memory_efficient_fp16=False, min_loss_scale=0.0001, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, only_source=False, optimizer='nag', padding_factor=8, seed=1, source_lang='ja', srcdict=None, target_lang='en', task='translation', tensorboard_logdir='', testpref='use_file/kyoto-test', tgtdict=None, threshold_loss_scale=None, thresholdsrc=0, thresholdtgt=0, tokenizer=None, trainpref='use_file/kyoto-train.cln', user_dir=None, validpref='use_file/kyoto-dev', workers=20)
| [ja] Dictionary: 114287 types
| [ja] use_file/kyoto-train.cln.ja: 329882 sents, 6415013 tokens, 0.0% replaced by <unk>
| [ja] Dictionary: 114287 types
| [ja] use_file/kyoto-dev.ja: 1166 sents, 28010 tokens, 0.678% replaced by <unk>
| [ja] Dictionary: 114287 types
| [ja] use_file/kyoto-test.ja: 1160 sents, 29638 tokens, 0.79% replaced by <unk>
| [en] Dictionary: 161663 types
| [en] use_file/kyoto-train.cln.en: 329882 sents, 6241368 tokens, 0.0% replaced by <unk>
| [en] Dictionary: 161663 types
| [en] use_file/kyoto-dev.en: 1166 sents, 25475 tokens, 2.15% replaced by <unk>
| [en] Dictionary: 161663 types
| [en] use_file/kyoto-test.en: 1160 sents, 27894 tokens, 1.98% replaced by <unk>
| Wrote preprocessed data to use_file/AT_bin

result
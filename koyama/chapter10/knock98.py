# 98. ドメイン適応
# Japanese-English Subtitle Corpus (JESC)やJParaCrawlなどの翻訳データを活用し，
# KFTTのテストデータの性能向上を試みよ．

'''
CUDA_VISIBLE_DEVICES=6 nohup fairseq-train \
  /work/aomi/100knock2020/chapter10/data/JESC/processed/bin \
  --save-dir /work/aomi/100knock2020/chapter10/knock98/models/model_1111/pretraining \
  --arch transformer \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt \
  --min-lr '1e-09' --warmup-init-lr '1e-07' \
  --warmup-updates 4000 \
  --dropout 0.3 \
  --max-epoch 10 \
  --clip-norm 1.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 1024 \
  --seed 1111 > train.log &
'''

'''
CUDA_VISIBLE_DEVICES=6 nohup fairseq-train \
  /work/aomi/100knock2020/chapter10/data/KFTT/processed/bin \
  --save-dir /work/aomi/100knock2020/chapter10/knock98/models/model_1111/fine_tuning \
  --arch transformer \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt \
  --min-lr '1e-09' --warmup-init-lr '1e-07' \
  --warmup-updates 4000 \
  --dropout 0.3 \
  --max-epoch 40 \
  --clip-norm 1.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 1024 \
  --seed 1111 > train.log &
'''
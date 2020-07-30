"""
# beam 1~20 prediction
!CUDA_VISIBLE_DEVICES=0 fairseq-interactive --path checkpoints/kftt.ja-en/checkpoint_best.pt --beam 1 data-bin/kftt.ja-en/ < ../kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94.1.out
!CUDA_VISIBLE_DEVICES=0 fairseq-interactive --path checkpoints/kftt.ja-en/checkpoint_best.pt --beam 5 data-bin/kftt.ja-en/ < ../kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94.5.out
!CUDA_VISIBLE_DEVICES=0 fairseq-interactive --path checkpoints/kftt.ja-en/checkpoint_best.pt --beam 10 data-bin/kftt.ja-en/ < ../kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94.10.out
!CUDA_VISIBLE_DEVICES=0 fairseq-interactive --path checkpoints/kftt.ja-en/checkpoint_best.pt --beam 15 data-bin/kftt.ja-en/ < ../kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94.15.out
!CUDA_VISIBLE_DEVICES=0 fairseq-interactive --path checkpoints/kftt.ja-en/checkpoint_best.pt --beam 20 data-bin/kftt.ja-en/ < ../kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94.20.out

# beam 1~20 evaluation
## beam 1
!fairseq-score --sys 94.1.out --ref ../kftt-data-1.0/data/tok/kyoto-dev.en
BLEU4 = 9.72, 37.7/13.8/6.0/2.9 (BP=1.000, ratio=1.162, syslen=28252, reflen=24309)
## beam 5
!fairseq-score --sys 94.5.out --ref ../kftt-data-1.0/data/tok/kyoto-dev.en
BLEU4 = 11.99, 41.8/16.3/7.7/3.9 (BP=1.000, ratio=1.055, syslen=25642, reflen=24309)
## beam 10
!fairseq-score --sys 94.10.out --ref ../kftt-data-1.0/data/tok/kyoto-dev.en
BLEU4 = 12.05, 41.8/16.5/7.8/3.9 (BP=1.000, ratio=1.046, syslen=25426, reflen=24309)
## beam 15
!fairseq-score --sys 94.15.out --ref ../kftt-data-1.0/data/tok/kyoto-dev.en
BLEU4 = 12.00, 41.5/16.4/7.8/3.9 (BP=1.000, ratio=1.046, syslen=25430, reflen=24309)
## beam 20
!fairseq-score --sys 94.20.out --ref ../kftt-data-1.0/data/tok/kyoto-dev.en
BLEU4 = 11.92, 41.2/16.3/7.8/3.9 (BP=1.000, ratio=1.049, syslen=25502, reflen=24309)
"""

import matplotlib.pyplot as plt
beam_sizes = [1, 5, 10, 15, 20]
bleus = [9.72, 11.99, 12.05, 12.00, 11.92]

plt.xlabel('beam-size')
plt.ylabel('bleu')
plt.plot(beam_sizes, belus)
plt.savefig('beam-bleu.png')
plt.show()


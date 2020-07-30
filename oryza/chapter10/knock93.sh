fairseq-score --sys knock92.out --ref test.de-en.en

# Namespace(ignore_case=False, order=4, ref='test.de-en.en', sacrebleu=False, sentence_bleu=False, sys='knock92.out')
# BLEU4 = 0.66, 15.7/1.1/0.2/0.1 (BP=0.913, ratio=0.917, syslen=120268, reflen=131156)
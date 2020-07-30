"""

# obtain dataset
!wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
!tar -zxvf ./kftt-data-1.0.tar.gz

# preprocess
!git clone https://github.com/pytorch/fairseq
!cd fairseq
!pip install --editable ./

!mkdir data-bin
!mkdir data-bin/kftt.ja-en
!TEXT=../kftt-data-1.0/data/tok
!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train \
    --validpref $TEXT/kyoto-dev \
    --testpref $TEXT/kyoto-test \
    --destdir data-bin/kftt.ja-en/ \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20

"""


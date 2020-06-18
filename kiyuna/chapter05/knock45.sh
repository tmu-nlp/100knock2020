#!/bin/sh
out=out45

if [ ! -e $out ]; then
    python knock45.py >$out
fi

# コーパス中で頻出する述語と格パターンの組み合わせ
cat $out | sort | uniq -c | sort -nr >${out}_freq

# 各同士の格パターン
for tgt in 行う なる 与える; do
    grep "^$tgt\s" $out | sort | uniq -c | sort -nr >${out}_$tgt
done

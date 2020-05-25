INPUT_PATH=./popular-names.txt

test() {
    N=$1
    rm out16*
    python knock16.py $INPUT_PATH $N
    num_line=$(cat $INPUT_PATH | wc -l)
    split -l $((($num_line + $N - 1) / $N)) $INPUT_PATH out16_
    wc -l out16a* | tail -n2 | head -n1 # 0 になっているか
    ls -1 out16_* | wc -l               # N と等しいか
}

# for i in $(seq 1 99); do
#     test $i
# done

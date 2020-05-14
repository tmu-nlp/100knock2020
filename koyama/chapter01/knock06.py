# 06. 集合
# “paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
# さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

from knock05 import n_gram

if __name__ == "__main__":
    str1 = "paraparaparadise"
    str2 = "paragraph"
    str3 = "se"
    X = n_gram(str1, t="char")
    X = set(X)
    Y = n_gram(str2, t="char")
    Y = set(Y)
    union = X | Y
    intersec = X & Y
    # 差集合演算は可換でないので、「XとYの差集合」という表現はよくない。
    diff1 = X - Y
    diff2 = Y - X
    print(f"{X=}")
    print(f"{Y=}")
    print(f"{union=}")
    print(f"{intersec=}")
    print(f"{diff1=}")
    print(f"{diff2=}")
    # 「XおよびYに含まれるかどうかを調べよ」は、XとYのそれぞれに対して別々に調べる、XとYの両方に含まれるかどうかを調べる、どっち？
    # 「および」は、文脈によって、「かつ」にも「または」にも言い換えられる？
    ans1 = str3 in X
    ans2 = str3 in Y
    ans3 = str3 in intersec
    print(f"{str3}はXに含まれるか：{ans1}")
    print(f"{str3}はYに含まれるか：{ans2}")
    print(f"{str3}はXとYの積集合に含まれるか：{ans3}")

# 実行結果
# X={'ar', 'ra', 'di', 'se', 'pa', 'ad', 'ap', 'is'}
# Y={'gr', 'ar', 'ra', 'pa', 'ag', 'ap', 'ph'}
# union={'gr', 'ar', 'ra', 'di', 'se', 'pa', 'ad', 'ph', 'ag', 'ap', 'is'}
# intersec={'ar', 'ra', 'ap', 'pa'}
# diff1={'se', 'is', 'ad', 'di'}
# diff2={'ag', 'gr', 'ph'}
# seはXに含まれるか：True
# seはYに含まれるか：False
# seはXとYの積集合に含まれるか：False
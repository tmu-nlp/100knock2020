# 07. テンプレートによる文生成Permalink
# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

def func(x, y, z):
    ans = f"{x}時の{y}は{z}"
    return ans

if __name__ == "__main__":
    ans = func(12, "気温", 22.4)
    print(f"{ans=}")

# 実行結果
# ans='12時の気温は22.4'
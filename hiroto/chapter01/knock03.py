#https://note.nkmk.me/python-split-strip-list-join/
#strip()は文字列の先頭・末尾の余分な文字を削除するメソッド。
str = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
words = [word.strip(',.') for word in str.split()] #','と'.'を除く
list = [len(word) for word in words]
print(list)

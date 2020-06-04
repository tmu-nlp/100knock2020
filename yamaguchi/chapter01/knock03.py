text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
word = text.replace(",", "").replace(".", "")

ans = [len(i) for i in word.split()]

print("各単語の文字数を並べたリストは，" + str(ans) + "です．\n")

S = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

list1 = S.split(" ")
list2 = []

for word in list1:
    list2.append(len(word) - word.count(",") - word.count("."))

print(list2)

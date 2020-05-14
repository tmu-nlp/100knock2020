sent = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
sent = sent.replace('.', '').replace(',', '').split(' ')

lengthList = []

for word in sent:
    lengthList.append(len(word))

print(lengthList)
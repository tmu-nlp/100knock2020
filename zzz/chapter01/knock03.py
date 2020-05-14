import re

s = 'Now I need a drink, alcoholic of course, ' \
    'after the heavy lectures involving quantum mechanics.'
s = re.sub(r',|\.', '', s)
# print(s)
words = s.split(' ')
# print(words)
res = []
for word in words:
    res.append(len(word))
print(res)

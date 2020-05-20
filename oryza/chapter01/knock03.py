"""
Knock03

Split the sentence “Now I need a drink, alcoholic of course, 
after the heavy lectures involving quantum mechanics.” 
into words, and create a list whose element presents 
the number of alphabetical letters in the corresponding word.
"""

sent = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
wordsp = sent.split()

print(wordsp)

count = []
for i in wordsp:
    c = len(i.strip('.,'))
    count.append(c)
    
print(count)

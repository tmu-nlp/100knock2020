wordA = 'パトカー'
wordB = 'タクシー'

wordCombined = ''

for i in range(len(wordA)):
    wordCombined = wordCombined + wordA[i] + wordB[i]

print(wordCombined)
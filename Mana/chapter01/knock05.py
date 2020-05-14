def word_n_gram(sent, n):
  wordNGramList = []
  wordsent = sent.split(' ')
  for i in range(len(wordsent)-n+1):
    wordNGramList.append(str(wordsent[i:i+n]))
  return set(wordNGramList)

def char_n_gram(sent, n):
  charNGramList = []
  charsent = sent.replace(' ', '')
  for i in range(len(charsent)-n+1):
    charNGramList.append(str(charsent[i:i+n]))
  return set(charNGramList)

sent = 'I am an NLPer'

print(word_n_gram(sent, 2))
print(char_n_gram(sent, 2))
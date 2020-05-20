"""
Knock02

Obtain the string “schooled” by concatenating the letters 
in “shoe” and “cold” one after the other from head to tail.
"""

word1 = 'shoe'
word2 = 'cold'

words = []
for i,j in zip(word1,word2):
    w = i + j
    words.append(w)

word_merge = ''.join(words)    
    
print(word_merge)

"""
Knock05

Implement a function that obtains n-grams from a given sequence object 
(e.g., string and list). Use this function to obtain word bi-grams and 
letter bi-grams from the sentence “I am an NLPer”
"""

sent = 'I am an NLPer'

def charNgram(sentence, n):
    chars = []
    for i in range(len(sentence)-1):
        c = sentence[i:i+n]
        chars.append(c)
    return chars

def wordNgram(sentence, n):
    words = []
    s = [word.strip(".,") for word in sentence.split()]
    for i in range(len(s)-1):
        c = s[i:i+n]
        words.append(c)
    return words
    
print(charNgram(sent,2))
print(wordNgram(sent,2))


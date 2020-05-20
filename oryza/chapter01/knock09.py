"""
Knock09

Write a program with the specification:

- Receive a word sequence separated by space
- For each word in the sequence:
  - If the word is no longer than four letters, keep the word unchanged
  Otherwise,
  - Keep the first and last letters unchanged
  - Shuffle other letters in other positions (in the middle of the word)

Observe the result by giving a sentence, e.g., “I couldn’t believe that I could actually 
understand what I was reading : the phenomenal power of the human mind .”
"""

import random

sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

def typoglycemia(sent):
    glyce = []
    words = sent.split(' ')
    for w in words:
        if len(w)<=4:
            temp = w
        else:
            toShuf = w[1:len(w)-1]
            shuffling = random.sample(toShuf,k=len(toShuf))
            temp = w[0] + ''.join(shuffling) + w[-1]
        glyce.append(temp)
    return ' '.join(glyce)

print(typoglycemia(sentence))

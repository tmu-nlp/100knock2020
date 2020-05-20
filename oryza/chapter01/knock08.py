"""
Knock08

Implement a function cipher that converts a given string with the specification:
- Every alphabetical letter c is converted to a letter 
  whose ASCII code is (219 - [the ASCII code of c])
- Keep other letters unchanged

Use this function to cipher and decipher an English message.
"""

sentence = input('Enter some strings: ')

def cipher(sent):
    cip = []
    for i in sent:
        if i == 'c':
            char = chr(219-ord('c'))
        elif i == chr(219-ord('c')):
            char = 'c'
        else:
            char = i
        cip.append(char)
    ciphered = ''.join(cip)
    return ciphered

def decipher(sent):
    decip = []
    for i in sent:
        if i == chr(219-ord('c')):
            char = 'c'
        elif i == 'c':
            char = chr(219-ord('c'))
        else:
            char = i
        decip.append(char)
    deciphered = ''.join(decip)
    return deciphered

print(cipher(sentence))
print(decipher(cipher(sentence)))

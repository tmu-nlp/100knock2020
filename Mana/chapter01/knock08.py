def cipher(word):
    wordCiphered = ''
    for i in range(len(word)):
        if word[i].islower():
            wordCiphered = wordCiphered + chr(219-ord(word[i]))
        else:
            wordCiphered = wordCiphered + word[i]
    return wordCiphered

def decipher(word):
    wordDeciphered = ''
    for i in range(len(word)):
        if word[i].islower():
            wordDeciphered = wordDeciphered + chr(219-ord(word[i]))
        else:
            wordDeciphered = wordDeciphered + word[i]
    return wordDeciphered

print(cipher('KomachiLabNlpKnock'))
print(decipher('KlnzxsrLzyNokKmlxp'))
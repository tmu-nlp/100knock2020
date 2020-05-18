def cipher(word):
    """
    """
    word_encrypted = ''
    for char in word:
        code = ord(char)
        if ord('a') <= code <= ord('z'):
            word_encrypted += chr(219 - code)
        else:
            word_encrypted += char
    return word_encrypted


def decrypt(word_encrypted):
    """
    """
    word_decrypted = ''
    for char in word_encrypted:
        code = ord(char)
        if ord('a') <= code <= ord('z'):
            word_decrypted += chr(219 - code)
        else:
            word_decrypted += char
    return word_decrypted


if __name__ == '__main__':
    word = 'GitHub'
    ans_encrypted = cipher(word)
    ans_decrypted = decrypt(ans_encrypted)
    
    print('plane: {}'.format(word))
    print('cipher: {}'.format(ans_encrypted))
    print('cipher: {}'.format(ans_decrypted))


def create_ngram(sentence, n):
    """ create n-gram
    :param sentence:
    :param n: int, n-gram

    :return ngram:
    """
    ngram = []
    for i in range(len(sentence) - n + 1):
        #print(i)
        ngram.append(sentence[i:i+n])
    return ngram

if __name__ == '__main__':
    sentence = 'I am an NLPer'

    print('word bi-gram:')
    ans_word_bigram = create_ngram(sentence.split(), 2)
    print(ans_word_bigram)

    print('character bi-gram:')
    ans_char_bigram = create_ngram(sentence, 2)
    print(ans_char_bigram)

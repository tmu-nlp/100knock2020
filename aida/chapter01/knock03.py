def del_punct(sentence):
    """ delete punctuation ',' '.'
    :param sentence: plane words list
    
    :return words_fixed: words deleted punctuations
    """
    sentence = sentence.replace(',', '').replace('.', '')
    words_fixed = sentence.split()
    return words_fixed

if __name__ == '__main__':
    sentence = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    words_fixed = del_puncts(sentence)
    ans = [len(word) for word in words_fixed]
    print(ans)


import random

def shuffle_sentence(sentence):
    """
    """
    random.seed(0)
    words = sentence.split()
    sentence_shuffled = []
    for word in words:
        if len(word) <= 4:
            sentence_shuffled.append(word)
        else:
            word_shuffled = ''
            word_shuffled += word[0]

            # shuffle word[1] ~ word[-2] characters
            chars = [char for char in word[1:-1]]
            random.shuffle(chars)
            word_shuffled += ''.join(chars)

            word_shuffled += word[-1]
            sentence_shuffled.append(word_shuffled)
    
    return ' '.join(sentence_shuffled)


if __name__ == '__main__':
    sentence = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
    ans = shuffle_sentence(sentence)
    print(ans)

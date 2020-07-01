from knock60 import load_model

if __name__ == '__main__':
    model = load_model()

    with open('./questions-words.txt') as fp:
        lines = fp.readlines()

    with open('./my-answer.txt', 'w') as fp:
        for line in lines:
            words = line.strip().split()                                     
            if len(words)==4:
                word, prob = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
                words.append(word)
                words.append(str(prob))
            fp.write(f"{' '.join(words)}\n")


from knock40 import Word
from knock41 import read_dependency

if __name__ == "__main__":
    jsonf = open('ai.en.txt.json','r') 
    text = read_dependency(jsonf)
    for sent in text:
        sentence = []
        for s in sent:
            dependency = s.print_dependency()
            dependency = dependency.split('\t')
            sentence.append(dependency[0])
            if dependency[2] == 'ROOT':
                print('Root Word: ' + dependency[0])
        print(' '.join(sentence) + '\n')
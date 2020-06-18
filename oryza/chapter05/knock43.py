from knock40 import Word
from knock41 import read_dependency

if __name__ == "__main__":
    jsonf = open('ai.en.txt.json','r') 
    text = read_dependency(jsonf)
    for sent in text:
        for s in sent:
            dependency = s.print_dependency()
            dependency = dependency.split('\t')
            if dependency[2][0] == 'n':
                print(dependency[1] + '\t' + dependency[2] + '\t' + dependency[0])
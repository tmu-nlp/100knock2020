import json
from knock40 import Word

def merged(dict_1,dict_2):
    sent_parse = []
    for i in range(len(dict_1)):
        words = Word(dict_1[i]['originalText'],dict_1[i]['lemma'],dict_1[i]['pos'],'','')
        head = ''
        dep = ''
        for j in range(len(dict_2)):
            if dict_1[i]['index'] == dict_2[j]['dependent']:
                head = dict_2[j]['governorGloss']
                dep = dict_2[j]['dep']
        child_list = []
        for k in range(len(dict_2)):
            if dict_1[i]['index'] == dict_2[k]['governor']:
                child_list.append(dict_2[k]['dependentGloss'])
        words.update_word(head,dep,child_list)
        sent_parse.append(words)
    return sent_parse

def read_dependency(json_file):
    data = json.load(json_file)
    text = data['sentences']
    sentences = []
    for sent in text:
        tokens = sent['tokens']
        deps = sent['basicDependencies']
        token_dep = merged(tokens,deps)
        sentences.append(token_dep)
    return sentences

if __name__ == "__main__":
    jsonf = open('ai.en.txt.json','r') 
    sent = read_dependency(jsonf)
    for s in sent[0]:
        print(s.print_dependency())
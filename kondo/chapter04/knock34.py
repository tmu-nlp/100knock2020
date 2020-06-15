from knock30 import data_mapping

analysis_file = "neko.txt.mecab"



def get_greedy_n():
    lines = data_mapping(analysis_file)
    t = 0
    longest_N_list = []
    Name_phrase = []
    for line in lines:
        for word in line:
            if word['pos'] == '名詞':
                Name_phrase.append(word['surface'])
            else:
                if len(Name_phrase) > 1:
                    longest_N_list.append("".join(Name_phrase))
                Name_phrase = []
        t += 1
        if t == 10:
            break
    return(longest_N_list)

if __name__ == "__main__":
    print(get_greedy_n())
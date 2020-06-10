from knock30 import data_mapping

analysis_file = "neko.txt.mecab"



def get_n_phrase():
    lines = data_mapping(analysis_file)
    t = 0
    N_phrase = []
    for line in lines:
        for i in range(1, len(line) - 1):
            if line[i]['surface'] == 'の' \
                    and line[i - 1]['pos'] == '名詞' \
                    and line[i + 1]['pos'] == '名詞':
                phrase = line[i-1]['surface'] + line[i]['surface'] + line[i+1]['surface']
                N_phrase.append(phrase)
        t += 1
        if t == 10:
            break
    return(N_phrase)

if __name__ == "__main__":
    print(get_n_phrase())
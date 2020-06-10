from knock30 import data_mapping

analysis_file = "neko.txt.mecab"



def get_v_sur():
    lines = data_mapping(analysis_file)
    t = 0
    V_sur = []
    for line in lines:
        for word in line:
            if word['pos'] == '動詞':
                surface = word['surface']
                V_sur.append(surface)
        t += 1
        if t == 5:
            break
    return(V_sur)

if __name__ == "__main__":
    print(get_v_sur())
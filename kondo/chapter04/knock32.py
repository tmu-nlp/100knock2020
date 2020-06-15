from knock30 import data_mapping

analysis_file = "neko.txt.mecab"



def get_v_base():
    lines = data_mapping(analysis_file)
    t = 0
    V_base = []
    for line in lines:
        for word in line:
            if word['pos'] == '動詞':
                base = word['base']
                V_base.append(base)
        t += 1
        if t == 5:
            break
    return(V_base)

if __name__ == "__main__":
    print(get_v_base())
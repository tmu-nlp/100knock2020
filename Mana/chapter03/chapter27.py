import re

f = open('britain.txt', 'r')
britain = f.readline()
basicInfo = re.search(r'\{\{基礎情報(.*?)\\n\}\}\\n', britain).group().replace("\'\'\'", "").replace("\'\'", "")
basicInfo2 = re.sub(r'\<ref\>(.*?)\<\/ref\>', " ", basicInfo).split("\\n")

dict_basic = {}

for i in range(len(basicInfo2)):
    info = basicInfo2[i].split("=")
    if len(info) == 2:
        if re.search(r'\{\{(.*?)\|(.*?)\}\}', info[1]):
            bracket = re.search(r'\{\{(.*?)\|[a-z]{2}\|(.*?)\}\}', info[1])
            info[1] = bracket.group(2)

        elif  re.search(r'\[\[(.*?)\|(.*?)\]\]', info[1]):
            if re.search(r'(\s|\w)*\]\]', info[1]):
                info[1] = (re.search(r'(\s|\w)*\]\]', info[1]).group()[:-2])

        info[1] = info[1].replace("]]", "").replace("[[", "")
        dict_basic[info[0][1:-1]] = info[1][1:]


for elem in dict_basic:
    print(elem+ ":" +dict_basic[elem])

f.close()

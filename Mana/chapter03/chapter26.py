import re

f = open('britain.txt', 'r')
britain = f.readline()

#deleting '' and '''
basicInfo = re.search(r'\{\{基礎情報(.*?)\\n\}\}\\n', britain).group().replace("\'\'\'", "").replace("\'\'", "")

basicInfo2 = re.sub(r'\<ref\>(.*?)\<\/ref\>', " ", basicInfo).split("\\n")

dict_basic = {}
for i in range(len(basicInfo2)):
  info = basicInfo2[i].split("=")
  if len(info) == 2:
    dict_basic[info[0][1:-1]] = info[1][1:]


for elem in dict_basic:
    print(elem+ ":" +dict_basic[elem])


f.close()
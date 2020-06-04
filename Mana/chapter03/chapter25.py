import re

f = open('britain.txt', 'r')
britain = f.readline()
basicInfo = re.search(r'\{\{基礎情報(.*?)\\n\}\}\\n', britain).group()
basicInfo2 = re.sub(r'\<ref\>(.*?)\<\/ref\>', " ", basicInfo).split("\\n")

dict_basic = {}
for i in range(len(basicInfo2)):
  info = basicInfo2[i].split("=")
  if len(info) == 2:
    dict_basic[info[0][1:]] = info[1]


for elem in dict_basic:
    print(elem+ ":" +dict_basic[elem])

"""
basicInforef = re.findall(r'\<ref\>(.*?)\<\/ref\>', basicInfo)
for match in basicInforef:
    match = match.split("\\n")
    for elem in match:
        print(elem)
"""


f.close()
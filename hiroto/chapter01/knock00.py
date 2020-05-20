str1 = 'stressed'
str2 = [];
for i in reversed(range(len(str1))):
    str2.append(str1[i])

str = ''.join(str2)
print(str)

#print(str1[::-1])

str1 = 'パトカー'
str2 = 'タクシー'
str = ''

for i in range(len(str1)):
    str += str1[i] + str2[i]

print(str)

'''
str_sub = ''
for (a, b) in zip(str1, str2):
    str_sub += a + b
print(str_sub)
'''

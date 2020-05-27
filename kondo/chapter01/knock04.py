S = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

num_tuple = (1, 5, 6, 7, 8, 9, 15, 16, 19)
list1 = S.split()
dict = {}

for num, word in enumerate(list1):
    if num + 1 in num_tuple:
        dict[word[0:1]] = num + 1
    else:
        dict[word[0:2]] = num + 1

print(dict)

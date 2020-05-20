"""
Knock04

Split the sentence “Hi He Lied Because Boron Could Not Oxidize Fluorine. 
New Nations Might Also Sign Peace Security Clause. Arthur King Can.” 
into words, and extract the first letter from the 1st, 5th, 6th, 7th, 
8th, 9th, 15th, 16th, 19th words and the first two letters from the other words. 
Create an associative array (dictionary object or mapping object) that maps from 
the extracted string to the position (offset in the sentence) of the corresponding word.
"""

sent = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
s = [word.strip(".,") for word in sent.split()]

atom = []
pos = []
for i in s:
    ind = s.index(i) + 1
    pos.append(ind)
    if ind in [1,5,6,7,8,9,15,16,19]:
        atom.append(i[0])
    else:
        atom.append(i[:2])

atomDict = {atom[i]:pos[i] for i in range(len(atom))}
print(atomDict)

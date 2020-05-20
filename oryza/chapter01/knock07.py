"""
Knock07

Implement a function that receives arguments, x, y, and z and 
returns a string “{y} is {z} at {x}”, where “{x}”, “{y}”, and “{z}” 
denote the values of x, y, and z, respectively. In addition, 
confirm the return string by giving the arguments x=12, y="temperature", and z=22.4.
"""

x = input('Enter time (hour/day/month/year): ')
y = input('Enter a word: ')
z = input('Enter a number: ')

print('\n' + y + ' is ' + z + ' at ' + x)

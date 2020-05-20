#https://code-schools.com/python-template/
import string

def template(x, y, z):
    str = '$hour時の$temperatureは$degree'
    s1 = string.Template(str)
    s2 = s1.substitute(hour = x, temperature = y, degree = z)
    return s2

print(template(12, '気温', 22.4))

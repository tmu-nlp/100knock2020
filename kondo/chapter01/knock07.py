def temp(x, y, z):
    x = str(x)
    z = str(z)
    return "{}時の{}は{}".format(x, y, z)

x = 12
y = "気温"
z = 22.4
print(temp(x, y, z))

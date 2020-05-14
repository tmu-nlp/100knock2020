def generate_message(x, y, z):
    """ generate message
    :param x, y, z:

    :return message:
    """
    message = '{}時の{}は{}'.format(x, y, z)
    return message

if __name__ == '__main__':
    x = 12
    y = '気温'
    z = 22.4
    ans = generate_message(x, y, z)
    print(ans)

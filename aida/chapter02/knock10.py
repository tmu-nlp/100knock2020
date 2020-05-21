# code
def read_file(file_path="./popular-names.txt"):
    with open(file_path) as fp:
        lines = fp.readlines()
        lines = [line.replace('\n', '') for line in lines]
    return lines

if __name__ == '__main__':
    lines = read_file()
    print('{}-lines'.format(len(lines)))

# unix command
"""
$wc -l popular-names.txt
    2780 popular-names.txt
"""

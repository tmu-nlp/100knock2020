with open('questions-words-ans.txt') as file:
    num = {'syn': 0, 'sem': 0}
    correct = {'syn': 0, 'sem': 0}
    status = None
    for line in file:
        if ':' in line:
            if 'gram' in line:
                status = 'syn'
            else:
                status = 'sem'
            continue

        context, predict = line.split('\t')
        ans = context.split(' ')[-1]
        predict = predict.replace('\n', '')

        num[status] += 1
        if ans == predict:
            correct[status] += 1
    print('semantic analogy accuracy: {0}'.format(float(correct['sem']) / num['sem']))
    print('syntactic analogy accuracy: {0}'.format(float(correct['syn']) / num['syn']))
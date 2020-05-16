import random

def randomize(sent):
  sent = sent.split(' ')
  randList = []
  for elem in sent:
    if len(elem) > 4:
      randList.append(elem[0] + ''.join(random.sample(list(elem[1:-1]), len(elem)-2)) + elem[-1])
    else:
      randList.append(elem)
  return ' '.join(randList)

sent = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

print(randomize(sent))
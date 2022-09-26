import random

a = [4, 6, 2, 1, 0, 5]


b = [i for i in range(len(a)) if a[i] > 2]
print(b)
b = [1]
c = random.choice(b)
print(c)
#print(a[c])

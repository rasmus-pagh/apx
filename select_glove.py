import numpy as np
import sys
import code

embedding = {}

def find_near(center, threshold):
    with open(center+"-"+str(threshold)+".txt", "w") as out:
        for word in embedding:
            if np.dot(embedding[word], embedding[center]) > threshold:
                v = ",".join([str(x) for x in embedding[word]])
                print(word+";"+v, file=out)

center = sys.argv[1]
threshold = float(sys.argv[2])
filename = "glove.twitter.27B.100d.txt"
f = open(filename, "r")

n = 0
for line in f:
    data = line.split(' ')
    word = data[0]
    vector = np.array([ float(data[i]) for i in range(1, len(data)) ])
    embedding[word] = vector / np.linalg.norm(vector)
    n += 1
    if n == 10000000: break

find_near(center, threshold)
code.interact(local=locals())

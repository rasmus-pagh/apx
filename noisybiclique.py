# Noisy biclique, 'noisybiclique.txt'
p = 0.4
n = 42
for i in range(n):
  for j in range(i+1, n):
    if i%2 == j%2:
      if np.random.random() < p:
        print(f"{i} {j}")
    else:
      if np.random.random() < 1 - p:
        print(f"{i} {j}")

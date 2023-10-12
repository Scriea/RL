import numpy as np
import matplotlib.pyplot as plt

V_p = [0.700000, 0.286720, 0.180000, 0.126000, 0.108000, 0.100000]
V_q = [0.080000, 0.126000, 0.200000, 0.300000, 0.400000]
p = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
q = [0.6,0.7,0.8,0.9,1]

#plt.figure(figsize=(12, 10))
plt.plot(p, V_p, marker='o')
plt.title("State Value vs p")
plt.xlabel("p")
plt.xticks(p)
plt.savefig("Graph 1.png")
plt.clf()


# plt.figure(figsize=(12, 10))
plt.plot(q, V_q, marker='o')
plt.title("State Value vs q")
plt.xticks(q)
plt.xlabel("q")
plt.savefig("Graph 2.png")
plt.clf()
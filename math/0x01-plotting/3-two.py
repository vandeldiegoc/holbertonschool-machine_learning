#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.xlim(0, 20000)
plt.ylim(0, 1)
lines = plt.plot(x, y1, 'r--', x, y2, 'g')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.legend(lines[:2], ['C-14', 'Ra-226'], loc='upper right')
plt.title('Exponential Decay of Radioactive Elements')
plt.savefig("two.png")
plt.show()

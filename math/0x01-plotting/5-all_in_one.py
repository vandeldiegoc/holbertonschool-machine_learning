#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure()
plt.suptitle('All in One')


plt.subplot(321)
plt.plot(y0, color='r')
plt.xlim(0, 10)

plt.subplot(322)
plt.scatter(x1, y1, color='magenta')
plt.xlabel('Height (in)', fontsize=8)
plt.ylabel('Weight (lbs)', fontsize=8)
plt.title("Men's Height vs Weight", fontsize=8)


plt.subplot(323,)
plt.plot(x2, y2)
plt.yscale('log')
plt.xlim(0, 28651)
plt.ylabel('Fraction Remaining', fontsize=8)
plt.xlabel('Time (years)', fontsize=8)
plt.title('Exponential Decay of C-14', fontsize=8)


plt.subplot(324)
plt.xlim(0, 20000)
plt.ylim(0, 1)
lines = plt.plot(x3, y31, 'r--', x3, y32, 'g')
plt.xlabel('Time (years)', fontsize=8)
plt.ylabel('Fraction Remaining', fontsize=8)
plt.legend(lines[:2], ['C-14', 'Ra-226'], loc='upper right')
plt.title('Exponential Decay of Radioactive Elements', fontsize=8)

plt.subplot(313)
plt.ylim(0, 30)
plt.xlim(0, 100)

plt.xlabel('Grades', fontsize=8)
plt.ylabel('Number of students', fontsize=8)
plt.title('Project A')
plt.hist(student_grades, bins=range(0, 100, 10), edgecolor="black")
plt.tight_layout()
plt.savefig("5-all_in_one.png")
plt.show()
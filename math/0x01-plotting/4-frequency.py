#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.ylim(0, 30)
plt.xlim(0, 100)
plt.xticks(range(0, 101, 10))
plt.xlabel('Grades')
plt.ylabel('Number of students')
plt.title('Project A')
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
plt.show()

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

fig = plt.figure(figsize=(8, 6))
layout = (3, 2)

line_graph = plt.subplot2grid(layout, (0, 0))
scatter = plt.subplot2grid(layout, (0, 1))
change_of_scale = plt.subplot2grid(layout, (1, 0))
two_is_better = plt.subplot2grid(layout, (1, 1))
frequency = plt.subplot2grid(layout, (2, 0), colspan=2)

line_graph.plot(y0, 'r-')
plt.xlim((0, 10))

scatter.scatter(x1, y1, c='m')
scatter.set_title("Men's Height vs Weight", fontsize='x-small')
scatter.set_xlabel("Height (in)", fontsize='x-small')
scatter.set_ylabel("Weight (lbs)", fontsize='x-small')

change_of_scale.plot(x2, y2)
change_of_scale.set_xlabel("Time (years)", fontsize='x-small')
change_of_scale.set_ylabel("Fraction Remaining", fontsize='x-small')
change_of_scale.set_title("Exponential Decay of C-14", fontsize='x-small')
change_of_scale.set_yscale(value="log")
change_of_scale.set_xlim((0, 28650))

two_is_better.plot(x3, y31, 'r--', label='C-14')
two_is_better.plot(x3, y32, 'g', label='Ra-226')
two_is_better.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
two_is_better.set_xlim((0, 20000))
two_is_better.set_ylim((0, 1))
two_is_better.set_xlabel('C-14', fontsize='x-small')
two_is_better.set_ylabel('Ra-226', fontsize='x-small')
two_is_better.legend(loc='upper right')

frequency.hist(student_grades, bins=10, edgecolor='black', range=(0, 100))
frequency.axis([0, 100, 0, 30])
frequency.set_xticks(np.arange(0, 110, step=10))
frequency.set_ylim((0, 30))
frequency.set_xlabel("Grades", fontsize='x-small')
frequency.set_ylabel("Number of Students", fontsize='x-small')
frequency.set_title("Project A", fontsize='x-small')

plt.suptitle("All in One")
plt.tight_layout()
plt.show()

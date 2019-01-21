#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
print(fruit)

x = np.arange(3)
width = 0.5
colors = ('r', 'y', '#ff8000', '#ffe5b4')
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]
data = (apples, bananas, oranges, peaches)
bottom = np.zeros(3)

for elem, color in zip(data, colors):
    plt.bar(x, elem, width, bottom=bottom, color=color, tick_label=('Farrah', 'Fred', 'Felicia'))
    bottom += elem
    plt.legend(labels=['apples', 'bananas', 'oranges', 'peaches'])

plt.ylabel("Quantity of Fruit")
plt.suptitle("Number of Fruit per Person")
plt.ylim((0, 80))
plt.show()

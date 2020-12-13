#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))


plt.yticks(np.arange(0, 90, 10))
plt.ylim(0,)
group = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
plt.ylabel("Quantity of Fruit", fontweight='bold')
plt.title("Number of Fruit per Person", fontweight='bold')


plt.bar(group, fruit[0], width=0.5, label=fruits[0], color='red') 
plt.bar(group, fruit[1], width=0.5, color='yellow', label=fruits[1], bottom=fruit[0])
plt.bar(group, fruit[2], width=0.5, color='#ff8000', label=fruits[2], bottom=fruit[1] + fruit[0])
plt.bar(group, fruit[3], width=0.5, color='#ffe5b4', label=fruits[3], bottom=fruit[2] + fruit[1] + fruit[0])
plt.legend()
plt.savefig("6-bars.png")
plt.show()
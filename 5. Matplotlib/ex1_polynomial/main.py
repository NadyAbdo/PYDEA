import numpy as np
import matplotlib.pyplot as plt

# Generate x values using numpy
x = np.linspace(-2, 2, 100)

f1 = x
f2 = x**2
f3 = x**3

plt.plot(x, f1, label='$x$', color='blue')
plt.plot(x, f2, label='$x^2$', color='orange')
plt.plot(x, f3, label='$x^3$', color='green')

plt.xlabel('x')
plt.ylabel('y')

plt.xlim(-2, 2)
plt.ylim(-4, 4)

plt.grid(True)

plt.legend()

plt.show()
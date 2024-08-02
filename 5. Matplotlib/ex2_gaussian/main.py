import numpy as np
import matplotlib.pyplot as plt

x0, y0, sigma = 0, 2, 2

x = np.linspace(-6, 6, 100)
y = np.linspace(-8, 8, 100)
x, y = np.meshgrid(x, y)
z = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the Gaussian distribution surface
ax.plot_surface(x, y, z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(-6, 6)
ax.set_ylim(-8, 8)
ax.set_zlim(0, 0.04)

ax.view_init(elev=25, azim=35)

plt.show()

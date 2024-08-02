import numpy as np
import math as m
import matplotlib.pyplot as plt

func = []
x_values = np.arange(0, 101, .01)
y_values = np.arange(0, 101, .01)
#range = np.arange(0, 1001, 1)

for i in range(len(x_values)) :
    if i == 0 :
        func.append(0)
    else :
        func.append(m.sqrt((x_values[i]*y_values[i])/(x_values[i]+y_values[i])))
#print(func)
'''plt.plot(x_values, func)
plt.show()

interp = np.interp(x_values, x_values, func)
plt.plot(x_values, func, 'o')
plt.plot(x_values, interp, '-x')
plt.show()

import numpy as np'''

degree = 2
func_poly = np.column_stack((x_values**2, x_values*y_values, y_values**2, x_values, y_values, np.ones_like(x_values)))
#print(func_poly)

coefficients = np.linalg.lstsq(func_poly, func, rcond=None)[0]

a, b, c, d, e, f = coefficients

print(f"The function is: {a:.7f} X² + {b:.7f} XY + {c:.7f} Y² + {d:.7f} X + {e:.7f} Y + {f:.7f} = 0")
'''func_test =[]
for i in range(len(x_values)) :
    func_test.append((a*(x_values[i]**2))+(b*x_values[i]*y_values[i])+(c*(y_values[i]**2))+(d*x_values[i])+(e*y_values[i])+(f))
print(func_test)'''

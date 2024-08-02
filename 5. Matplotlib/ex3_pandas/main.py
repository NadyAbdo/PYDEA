import pandas as pd
import numpy as np

x_values = np.arange(-5, 5.5, 0.5)
y_values = np.arange(-5, 5.5, 0.5)
x, y = np.meshgrid(x_values, y_values)
z = np.exp(x - y)

# Flatten the arrays
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

df = pd.DataFrame({'x': x_flat, 'y': y_flat, 'z': z_flat})
#print(df)
df.to_csv('./output.csv', index=False)

df_modified = pd.read_csv('./output.csv')
df_modified['z²'] = df_modified['z'] ** 2
#print(df_modified)
df_modified.to_csv('./output_modified.csv', index=False)
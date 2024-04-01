import numpy as np

# Example 2D arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6]])
array3 = np.array([[9, 10], [11, 12]])

# Stack arrays along a new axis (axis 0)
stacked_array = np.stack((array1, array2, array3), axis=0)

print("Stacked 3D array:")
print(stacked_array)
import numpy as np
import PixelEncoder as pe

arr = np.array([[2, 3], [3, 4], [6, 6], [3, 2]])

arr2 = arr[:, 0]

arr = pe.pixelToPoint(arr, 6, 6)

print(arr)

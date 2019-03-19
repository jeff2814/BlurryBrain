import numpy as np
import struct
import matplotlib.pyplot as plt

with open('train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))
    
    
with open ('train-labels-idx1-ubyte', 'rb') as g:
    magic, size = struct.unpack(">II", g.read(8))
    data1 = np.fromfile(g, dtype=np.dtype(np.uint8).newbyteorder('>'))
    
print(data1[4444]);
plt.imshow(data[4444,:,:], cmap='gray')
plt.show()
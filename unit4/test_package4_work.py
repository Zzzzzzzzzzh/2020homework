from pyspherepack import Box
import numpy as np

b = Box(10,box=np.array([1.0,1.0]),n_iters=60000)
b.pack()
b.plot()

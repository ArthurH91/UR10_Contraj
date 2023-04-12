import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

z = 2*np.ones((10,1))

x = np.eye(10)

x = np.concatenate((z.T,x))

y = np.eye(10,k=-1)

plt.subplot(121)
plt.imshow(x, interpolation='nearest', cmap=cm.Greys_r)
plt.colorbar()
plt.title("Hessian matrix")
plt.subplot(122)
plt.imshow(y, interpolation='nearest', cmap = cm.Greys_r)
plt.title("Derivative residuals")
plt.colorbar()
plt.show()
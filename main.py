import numpy as np
import matplotlib.pylab as plt

from solveFDFD import solveFDFD

pol = 'TM'
L0 = 1e-6
wvlen = 0.5
xrange = [-2, 2]
yrange = [-2, 2]
Nx = 100
Ny = 100
eps_r = np.ones((Nx,Ny))
eps_r[20:40, 20:80] = 3
SRC = np.zeros((Nx,Ny))
SRC[Nx//2,Ny//2] = 1
Npml = [10,10,10,10]
timing = True
(Ex, Ey, Ez, Hx, Hy, Hz, omega) = solveFDFD(pol, L0, wvlen, xrange, yrange, eps_r, SRC, Npml,timing)

plt.imshow(np.real(Ez))
plt.show()

#plt.imshow(SRC)
#plt.show()
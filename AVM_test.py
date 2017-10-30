import numpy as np
import matplotlib.pylab as plt

from solveFDFD import solveFDFD

pol = 'TM'
timing = True
L0 = 1e-6
wvlen = 0.5
xrange = [-2, 2]
yrange = [-2, 2]
Nx = 200
Ny = 100
L_box = 50
H_box = 40
eps = 3
npml = 10

nx = Nx//2
ny = Ny//2
src_pos_og = (Nx/2.0-npml)//2 + npml
src_pos_aj = Nx - (Nx/2.0-npml)//2 + npml
eps_r = np.ones((Nx,Ny))
eps_r[nx-L_box//2:nx+L_box//2, ny-H_box//2:ny+H_box//2] = eps
SRC_og = np.zeros((Nx,Ny))
SRC_aj = np.zeros((Nx,Ny))
SRC_og[src_pos_og,ny] = 1
SRC_aj[src_pos_og,ny] = 1
Npml = [npml,npml,npml,npml]

(_, _, Ez_og, Hx_og, Hy_og, _, _) = solveFDFD(pol, L0, wvlen, xrange, yrange, eps_r, SRC_og, Npml, timing=timing)

plt.imshow(np.real(Ez_og))
plt.show()

#plt.imshow(SRC)
#plt.show()
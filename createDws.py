import numpy as np
import scipy.sparse as sp
import matplotlib.pylab as plt

from time import time

def createDws(w, s, dL, N):
    # w: one of 'x', 'y', 'z'
    # s: one of 'f' and 'b'
    # dL: [dx dy dz] for 3D [dx dy] for 2D
    # N: [Nx Ny Nz] for 3D [Nx Ny] for 2D
    # note: this only does 2D simulations!!

    if w == 'x':
        dw = dL[0]
    elif w == 'y':
        dw = dL[1]
    else:
        raise ValueError, 'error reading in w type of Dws, must be one of {x,y}'        
        

    sign = 0
    if (s == 'f'):
        sign = -1
    elif (s == 'b'):
        sign = 1
    else:
        raise ValueError, 'error reading in s type of Dws, must be one of {f,b}'

    M = np.prod(N)  # total number of cells in domain

    # Compute Nx, Ny, and Nz
    Nx = N[0] 
    Ny = N[1] 
    Dws = sp.lil_matrix((Nx*Ny,Nx*Ny))

    # Check the direction of the derivative matrix to be created
    if (w == 'x'):
        # Construct a block of the derivative matrix in the x-direction
        tmp = sp.eye(Nx, format='csr')
        block_x = -tmp + sp.hstack((tmp[:,1:],tmp[:,:1]),format='csr').T
        # Create Dws = Dxs by stacking up the blocks
        for n in range(Ny):
            Dws[n*Nx : (n+1)*Nx, n*Nx: (n+1)*Nx] = 1/dw * block_x 

    if (w == 'y'):
        # Construct the block for Dys

        block_y = sp.lil_matrix((Nx*Ny,Nx*Ny))

        for n in range(Ny):
            block_y[n*Nx : (n+1)*Nx, n*Nx:(n+1)*Nx] = -sp.eye(Nx)

            if (n < Ny - 1):
                block_y[n*Nx : (n+1)*Nx, (n+1)*Nx : (n+2)*Nx] = sp.eye(Nx) 
            else:
                block_y[n*Nx : (n+1)*Nx, 0 : Nx] = sp.eye(Nx) 

        Dws = 1/dw*block_y
        
    # Check if the matrix needs to be transposed for backwards matrices
    if sign == -1:
        Dws = -Dws.T

    return Dws

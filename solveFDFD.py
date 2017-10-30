import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pylab as plt

from time import time
from S_create import S_create
from createDws import createDws
from bwdmean_w import bwdmean_w

def solveFDFD(pol, L0, wvlen, xrange, yrange, eps_r, SRC, Npml, timing=False):
    ## Input Parameters
    # L0: length unit (e.g., L0 = 1e-9 for nm)
    # wvlen: wavelength in L0
    # xrange: [xmin xmax], range of domain in x-direction including PML
    # yrange: [ymin ymax], range of domain in y-direction including PML
    # eps_r: Nx-by-Ny array of relative permittivity
    # SRC: Nx-by-Ny array of current source density (electric or magnetic)
    # Npml: [Nx_pml Ny_pml], number of cells in x- and y-normal PML

    ## Output Parameters
    # Ez, Hx, Hy: Nx-by-Ny arrays of H- and E-field components
    # dL: [dx dy] in L0
    # A: system matrix of A x = b
    # omega: angular frequency for given wvlen

    ## Set up the domain parameters.
    if timing: t = time()
    eps0 = 8.854e-12 * L0  # vacuum permittivity in farad/L0
    mu0 = np.pi * 4e-7 * L0  # vacuum permeability in henry/L0
    c0 = 1/np.sqrt(eps0*mu0)  # speed of light in vacuum in L0/sec

    N = eps_r.shape                   # [Nx Ny]
    L = [np.diff(xrange), np.diff(yrange)]  # [Lx Ly]
    dL = np.true_divide(L, N)[0]                # [dx dy]
    (Nx,Ny) = N

    if eps_r.shape != SRC.shape:
        raise ValueError, 'permittivity grid (eps_r) and current source (SRC) must be of the same size'   
         
    M = np.prod(N) 

    omega = 2*np.pi*c0/wvlen  # angular frequency in rad/sec

    if timing: print('setup time          : ' + str( time()-t) )
    
    ## Deal with the s_factor    
    if timing: t = time()
    (Sxf, Sxb, Syf, Syb) = S_create(L0, wvlen, xrange, yrange, N, Npml) 
    if timing: print('S-parameters        : '  + str( time()-t))
    
    ## Construct derivate matrices with PML
    if timing: t = time()
    Dyb = Syb*createDws('y', 'b', dL, N)
    Dxb = Sxb*createDws('x', 'b', dL, N)
    Dxf = Sxf*createDws('x', 'f', dL, N)
    Dyf = Syf*createDws('y', 'f', dL, N)
    if timing: print('derivative matrices : ' + str(time()-t))

    ## Average the epsilon space and create the epsilon tensors 
    eps_x = bwdmean_w(eps0 * eps_r, 'x')
    eps_y = bwdmean_w(eps0 * eps_r, 'y') 
    eps_z = eps0 * eps_r 

    vector_eps_x = np.reshape(eps_x.T, (M, 1)) 
    vector_eps_y = np.reshape(eps_y.T, (M, 1))
    vector_eps_z = np.reshape(eps_z.T, (M, 1))

    T_eps_x = sp.spdiags(vector_eps_x.T, 0, M, M) 
    T_eps_y = sp.spdiags(vector_eps_y.T, 0, M, M) 
    T_eps_z = sp.spdiags(vector_eps_z.T, 0, M, M) 


    ## Construct solver for either TE or TM polarization
    ## TE: Hz, Ex, Ey
    ## TM: Ez, Hx, Hy
    Ez = [] 
    Hx = [] 
    Hy = [] 
    Ex = [] 
    Ey = [] 
    Hz = [] 

    if timing: t = time()    
    if (pol == 'TE'):
        ## Construct A matrix and b vector
        A = Dxf * 1./T_eps_x * Dxb + Dyf * 1./T_eps_y * Dyb + (omega**2) * mu0 * sp.eye(M)
        mz = np.reshape(SRC, (M, 1))
        b = 1j*omega*mz 
        
        ## Solve the system of equations
        hz = spl.spsolve(A,b)
        
        ## Find electric fields
        ex = -1j/omega * 1./T_eps_y * Dyb * hz
        ey =  1j/omega * 1./T_eps_x * Dxb * hz       
        
        ## Return field matrices
        Hz = np.reshape(hz, (Nx,Ny))
        Ex = np.reshape(ex, (Nx,Ny))
        Ey = np.reshape(ey, (Nx,Ny))

    elif (pol == 'TM'):
        ## Construct A matrix and b vector
        A = Dxb * 1./mu0 * Dxf + Dyb * 1./mu0 * Dyf + (omega**2)*T_eps_z

        jz = np.reshape(SRC, M, 1) 
        b = 1j * omega * jz 
        
        ## Solve the system of equations
        ez = spl.spsolve(A,b)
        
        ## Find magnetic fields
        hx = -1/(1j*omega) * 1./mu0 * Dyb * ez
        hy =  1/(1j*omega) * 1./mu0 * Dxb * ez

        ## Return field matrices
        Ez = np.reshape(ez, (Ny,Nx))
        Hx = np.reshape(hx, (Ny,Nx))
        Hy = np.reshape(hy, (Ny,Nx))
        
        
    else:
        raise ValueError, 'Invalid polarization. Please specify either TE or TM' 
        
    if timing: print('final system solving: ' + str(time() - t))
    return (Ex, Ey, Ez, Hx, Hy, Hz, omega)

import numpy as np
import scipy.sparse as sp

from create_sfactor import create_sfactor

def S_create(L0, wvlen, xrange, yrange, N, Npml):
    #S_CREATE Summary of this function goes here
    #   Detailed explanation goes here


    ## Set up the domain parameters.
    eps0 = 8.854e-12 * L0  # vacuum permittivity in farad/L0
    mu0 = np.pi * 4e-7 * L0  # vacuum permeability in henry/L0
    c0 = 1/np.sqrt(eps0*mu0)  # speed of light in vacuum in L0/sec

    # L = [diff(xrange) diff(yrange)]  # [Lx Ly]
    # dL = L./N  # [dx dy]

    M = np.prod(N) 

    omega = 2*np.pi*c0/wvlen  # angular frequency in rad/sec

    ## Deal with the s_factor
    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor(xrange, 'f', omega, eps0, mu0, N[0], Npml[0]) 
    s_vector_x_b = create_sfactor(xrange, 'b', omega, eps0, mu0, N[0], Npml[0]) 
    s_vector_y_f = create_sfactor(yrange, 'f', omega, eps0, mu0, N[1], Npml[1]) 
    s_vector_y_b = create_sfactor(yrange, 'b', omega, eps0, mu0, N[1], Npml[1]) 


    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(N,dtype=complex) 
    Sx_b_2D = np.zeros(N,dtype=complex)
    Sy_f_2D = np.zeros(N,dtype=complex)
    Sy_b_2D = np.zeros(N,dtype=complex)

    for j in range(1,N[1]+1):
        Sx_f_2D[:, j-1] = s_vector_x_f[:,0] ** -1  
        Sx_b_2D[:, j-1] = s_vector_x_b[:,0] ** -1     

    for i in range(1,N[0]+1):
        Sy_f_2D[i-1, :] = s_vector_y_f[:,0] ** -1 
        Sy_b_2D[i-1, :] = s_vector_y_b[:,0] ** -1 

    # surf(abs(Sy_f_2D)) pause

    # Reshape the 2D s-factors into a 1D s-array
    Sx_f_vec = np.reshape(Sx_f_2D, M, 1) 
    Sx_b_vec = np.reshape(Sx_b_2D, M, 1) 
    Sy_f_vec = np.reshape(Sy_f_2D, M, 1) 
    Sy_b_vec = np.reshape(Sy_b_2D, M, 1) 

    # Construct the 1D total s-array into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, M, M) 
    Sx_b = sp.spdiags(Sx_b_vec, 0, M, M) 
    Sy_f = sp.spdiags(Sy_f_vec, 0, M, M) 
    Sy_b = sp.spdiags(Sy_b_vec, 0, M, M) 


    return (Sx_f, Sx_b, Sy_f, Sy_b)

import numpy as np

def create_sfactor(wrange, s, omega, eps0, mu0, Nw, Nw_pml):
    ## Input Parameters
    # wrange: [wmin wmax], range of domain in w-direction including PML
    # s: 'b' or 'f', indicating whether s-factor is for Dwb or Dwf
    # omega: angular frequency
    # eps0: vacuum permittivity
    # mu0: vacuum permeability
    # Nw: number of cells in w-direction
    # Nw_pml: number of cells in PML

    ## Output Parameter
    # sfactor_array: 1D array with Nw elements containing PML s-factors for Dws

    eta0 = np.sqrt(mu0/eps0)  # vacuum impedance
    m = 4  # degree of polynomial grading
    lnR = -12  # R: target reflection coefficient for normal incidence

    # find dw 
    hw = np.true_divide(np.diff(wrange),(Nw))
    dw = Nw_pml * hw

    # Sigma function
    sig_max = -(m+1) * lnR * 1.0 / (2.0*eta0*dw) 
    sig_w = lambda l: sig_max*(l*1.0/dw)**m 
    
    S = lambda l: 1 - 1j*sig_w(l) / (omega*eps0) 

    ## PML vector
    # initialize sfactor_array
    sfactor_array = np.ones((Nw, 1), dtype=complex)
    for i in range(1,Nw+1):
        if s == 'f':
            if (i <= Nw_pml):
                sfactor_array[i-1] = S(hw * (Nw_pml - i + 0.5)) 
                
            elif i > Nw - Nw_pml:
                sfactor_array[i-1] = S(hw * (i - (Nw - Nw_pml) - 0.5)) 
                    
        elif s == 'b':
            if i <= Nw_pml:
                sfactor_array[i-1] = S(hw * (Nw_pml - i + 1)) 
                
            elif i > Nw - Nw_pml:
                sfactor_array[i-1] = S(hw * (i - (Nw - Nw_pml) - 1)) 
        else:
            print('wrong s-factor char (must be b or f)')

    return sfactor_array
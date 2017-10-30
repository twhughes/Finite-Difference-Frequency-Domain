import numpy as np

def bwdmean_w(center_array, w):

    ## Input Parameters
    # center_array: 2D array of values defined at cell centers
    # w: 'x' or 'y', direction in which average is taken

    ## Out Parameter
    # avg_array: 2D array of averaged values
    if w == 'y':    
        center_shifted = np.roll(center_array, -1,axis=1)
    elif w == 'x':
        center_shifted = center_array
    else:
        raise ValueError, 'specified bwdmean_w element must be one of {x,y}'    
    avg_array = (center_shifted + center_array) / 2;
    return avg_array
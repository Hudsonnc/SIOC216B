import numpy as np 

def f_ij(x_j, mean_x_j, var_x_j, alpha_i, beta_i, cstar_ij):
    c_ij = alpha_i * var_x_j + beta_i * np.abs(mean_x_j)
    
    if np.abs(x_j) < c_ij and np.abs(mean_x_j) < cstar_ij:
        return mean_x_j 
    else:
        return(x_j)
    
def w_ij(var_x_j, cov_ij, a_i, b_i):
    return a_i * var_x_j + b_i * cov_ij
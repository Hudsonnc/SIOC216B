import numpy as np 

#def f_ij(x_j, mean_x_j, var_x_j, alpha_i, beta_i, c_ij, cstar_ij):
#    if np.abs(x_j) < c_ij and np.abs(mean_x_j) < cstar_ij:
#        return mean_x_j 
#    else:
#        return(x_j)
    
    
    
def f_ij(x_j, mean_x_j, var_x_j, alpha_i, beta_i, c_ij, cstar_ij):
    if not np.abs(x_j) < c_ij and np.abs(mean_x_j) < cstar_ij:
        return((x_j, [1,0]))
    elif np.abs(x_j) < c_ij and not np.abs(mean_x_j) < cstar_ij:
        return((x_j, [0,1]))
    elif not np.abs(x_j) < c_ij and not np.abs(mean_x_j) < cstar_ij:
        return((x_j, [1,1]))
    else:
        return((mean_x_j, [0,0]))   
    
    
    
    
def w_ij(var_x_j, cov_ij, a_i, b_i):
    return a_i * var_x_j + b_i * cov_ij
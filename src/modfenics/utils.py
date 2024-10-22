import torch
import dolfin as df
from scimba.equations.domain import SpaceTensor
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_test_sample_fromV(V_test,params):
    # get coordinates of the dof
    XXYY = V_test.tabulate_dof_coordinates()
    X_test = torch.tensor(XXYY,requires_grad=True)
    X_test = SpaceTensor(X_test,torch.zeros_like(X_test,dtype=int))

    # get parameters
    nb_params = len(params)
    shape = (XXYY.shape[0],nb_params)
    ones = torch.ones(shape)
    mu_test = (torch.Tensor(params).to(device) * ones).to(device)

    return X_test,mu_test

def get_utheta_fenics_onV(V_test,params,u_PINNs):
    X_test,mu_test = get_test_sample_fromV(V_test,params)
    
    pred = u_PINNs.setup_w_dict(X_test, mu_test)
    phi_tild = pred["w"][:,0].cpu().detach().numpy()
    
    u_theta = df.Function(V_test)
    u_theta.vector()[:] = phi_tild.copy()
    
    return u_theta
    
def get_laputheta_fenics_fromV(V_test,params,u_PINNs):
    X_test,mu_test = get_test_sample_fromV(V_test,params)
    
    pred = u_PINNs.setup_w_dict(X_test, mu_test)
    u_PINNs.get_first_derivatives(pred, X_test)
    u_PINNs.get_second_derivatives(pred, X_test)
    
    phi_tild_xx = pred["w_xx"][:,0].cpu().detach().numpy()
    phi_tild_yy = pred["w_yy"][:,0].cpu().detach().numpy()
    lap_phi_tild = phi_tild_xx + phi_tild_yy
    
    lapu_theta = df.Function(V_test)
    lapu_theta.vector()[:] = lap_phi_tild.copy()
    
    return lapu_theta

def get_param(i,parameter_domain):
    # pick 1 random parameter
    np.random.seed(0)
    for j in range(i):
        param = [np.random.uniform(parameter_domain[0][0], parameter_domain[0][1]), np.random.uniform(parameter_domain[1][0], parameter_domain[1][1])]
    param = np.round(param, 2)
    return param

def compute_slope(i,tab_nb_vert,tab_err):
    start = [tab_nb_vert[i],tab_err[i]]
    end = [tab_nb_vert[i-1],tab_err[i-1]]
    third = [end[0],start[1]]

    tri_x = [end[0], third[0], start[0], end[0]]
    tri_y = [end[1], third[1], start[1], end[1]
    ]
    plt.plot(tri_x, tri_y, "k--", linewidth=0.5)

    slope = -(np.log(start[1])-np.log(end[1]))/(np.log(start[0])-np.log(end[0]))
    slope = slope.round(2)
    
    vert_mid = [(end[0]+third[0])/2., (end[1]+third[1])/2.]
    
    return slope,vert_mid
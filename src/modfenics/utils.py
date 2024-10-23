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

def get_gradutheta_fenics_fromV(V_test,params,u_PINNs):
    X_test,mu_test = get_test_sample_fromV(V_test,params)
    
    pred = u_PINNs.setup_w_dict(X_test, mu_test)
    u_PINNs.get_first_derivatives(pred, X_test)
    
    phi_tild_x = pred["w_x"][:,0].cpu().detach().numpy()
    phi_tild_y = pred["w_y"][:,0].cpu().detach().numpy()
    
    # u_theta_x = df.Function(V_test)
    # u_theta_x.vector()[:] = phi_tild_x.copy()
    
    # u_theta_y = df.Function(V_test)
    # u_theta_y.vector()[:] = phi_tild_y.copy()
    
    V_test_2D = df.VectorFunctionSpace(V_test.mesh(),V_test.ufl_element().family(),V_test.ufl_element().degree(),dim=2)
    grad_utheta = df.Function(V_test_2D)
    grad_utheta.sub(0).vector()[:] = phi_tild_x.copy()
    grad_utheta.sub(1).vector()[:] = phi_tild_y.copy()
    
    return grad_utheta

def get_divmatgradutheta_fenics_fromV(V_test,params,u_PINNs,anisotropy_matrix):
    X_test,mu_test = get_test_sample_fromV(V_test,params)
    
    pred = u_PINNs.setup_w_dict(X_test, mu_test)
    u_PINNs.get_first_derivatives(pred, X_test)
    
    phi_tild_x = pred["w_x"][:,0]
    phi_tild_y = pred["w_y"][:,0]
    
    m00,m01,m10,m11 = anisotropy_matrix(torch,X_test.x.T,params)
    
    matgrad_x = m00 * phi_tild_x + m01 * phi_tild_y
    matgrad_y = m10 * phi_tild_x + m11 * phi_tild_y
    
    ones = torch.ones_like(matgrad_x)
    mat_grad_xx,_ = torch.autograd.grad(matgrad_x, X_test.x, ones, create_graph=True)[0].T
    
    ones = torch.ones_like(matgrad_y)
    _,mat_grad_yy = torch.autograd.grad(matgrad_y, X_test.x, ones, create_graph=True)[0].T
    
    divmatgradphitild = (mat_grad_xx + mat_grad_yy).cpu().detach().numpy()
    
    divmatgradutheta = df.Function(V_test)
    divmatgradutheta.vector()[:] = divmatgradphitild.copy()
    
    return divmatgradutheta

def get_param(i,parameter_domain):
    # pick 1 random parameter
    np.random.seed(0)
    for j in range(i):
        param = []
        for k in range(len(parameter_domain)):
            param.append(np.random.uniform(parameter_domain[k][0], parameter_domain[k][1]))
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
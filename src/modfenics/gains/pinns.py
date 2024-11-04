import pandas as pd
import os
import numpy as np
from testcases.utils import create_tree,get_random_params,compute_slope
from modfenics.error_estimations.utils import get_solver_type
from modfenics.utils import get_utheta_fenics_onV
import matplotlib.pyplot as plt

def read_csv_PINNs(csv_file):
    df_PINNs = pd.read_csv(csv_file)
    tab_nb_vert_PINNs = df_PINNs.values[0,1:]
    tab_h_PINNs = df_PINNs.values[1,1:]
    tab_err_PINNs = df_PINNs.values[2:,1:]
    return df_PINNs,tab_nb_vert_PINNs,tab_h_PINNs,tab_err_PINNs

def compute_error_pinns_deg(n_params,problem,degree,high_degree,u_theta,error_degree=4,new_run=False,result_dir="./"):
    dim = problem.dim
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = get_random_params(n_params,parameter_domain)   
    solver_type = get_solver_type(dim,testcase,version)
    
    save_uref = None
    # if not problem.ana_sol:
    #     savedir = result_dir + "u_ref/"
    #     create_tree(savedir)
    #     filename = savedir + f"u_ref_{param_num}.npy"
    #     save_uref = [filename]
    #     print(filename)   
    
    csv_file = result_dir+f'PINNs_errors_case{testcase}_v{version}_degree{degree}.csv'
    if not new_run and os.path.exists(csv_file):
        print(f"## Read csv file {csv_file}")
        df_PINNs, tab_nb_vert_PINNs, tab_h_PINNs, tab_err_PINNs = read_csv_PINNs(csv_file)
    else:
        print(f"## Run gains with PINNs for degree={degree}")
        tab_nb_vert_PINNs = [20,40]
        tab_h_PINNs = []
        tab_err_PINNs = np.zeros((n_params,len(tab_nb_vert_PINNs)))
        
        solver = solver_type(params=params, problem=problem, degree=degree, error_degree=error_degree, high_degree=high_degree, save_uref=save_uref)
        for (j,nb_vert) in enumerate(tab_nb_vert_PINNs):
            print(f"nb_vert={nb_vert}")
            solver.set_meshsize(nb_cell=nb_vert-1)            
            tab_h_PINNs.append(np.round(solver.h,3))
            
            for i in range(n_params):
                print(i,end=" ")
                norme_L2 = solver.pinns(i,u_theta)
                tab_err_PINNs[i,j] = norme_L2
            
        col_names = [("PINNs",str(tab_nb_vert_PINNs[i]),tab_h_PINNs[i]) for i in range(len(tab_nb_vert_PINNs))]
        mi = pd.MultiIndex.from_tuples(col_names, names=["method","n_vert","h"])
        df_PINNs = pd.DataFrame(tab_err_PINNs,columns=mi)
        df_PINNs.to_csv(csv_file)
        
        df_PINNs = pd.DataFrame(tab_err_PINNs,columns=mi)
        df_PINNs.to_csv(csv_file)

    return df_PINNs, tab_nb_vert_PINNs, tab_h_PINNs, tab_err_PINNs

def compute_error_pinns_all(param_num,problem,high_degree,error_degree=4,new_run=False,result_dir="./"):
    for d in [1, 2, 3]:
        _, _, _, _ = compute_error_pinns_deg(param_num,problem,d,high_degree,error_degree=error_degree,new_run=new_run,result_dir=result_dir)    
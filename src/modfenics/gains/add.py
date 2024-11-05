import pandas as pd
import os
import numpy as np
from testcases.utils import create_tree,get_random_params,compute_slope
from modfenics.error_estimations.utils import get_solver_type
from modfenics.utils import get_utheta_fenics_onV
import matplotlib.pyplot as plt

def read_csv_Corr(csv_file):
    df_Corr = pd.read_csv(csv_file)
    tab_nb_vert_Corr = df_Corr.values[0,1:]
    tab_h_Corr = df_Corr.values[1,1:]
    tab_err_Corr = df_Corr.values[2:,1:]
    return df_Corr,tab_nb_vert_Corr,tab_h_Corr,tab_err_Corr

def compute_error_Corr_deg(n_params,problem,degree,high_degree,u_theta,error_degree=4,new_run=False,result_dir="./"):
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
    
    csv_file = result_dir+f'Corr_errors_case{testcase}_v{version}_degree{degree}.csv'
    if not new_run and os.path.exists(csv_file):
        print(f"## Read csv file {csv_file}")
        df_Corr, tab_nb_vert_Corr, tab_h_Corr, tab_err_Corr = read_csv_Corr(csv_file)
    else:
        print(f"## Run gains with Corr for degree={degree}")
        tab_nb_vert_Corr = [20,40]
        tab_h_Corr = []
        tab_err_Corr = np.zeros((n_params,len(tab_nb_vert_Corr)))
        
        solver = solver_type(params=params, problem=problem, degree=degree, error_degree=error_degree, high_degree=high_degree, save_uref=save_uref)
        for (j,nb_vert) in enumerate(tab_nb_vert_Corr):
            print(f"nb_vert={nb_vert}")
            solver.set_meshsize(nb_cell=nb_vert-1)            
            tab_h_Corr.append(np.round(solver.h,3))
            
            for i in range(n_params):
                print(i,end=" ")
                _,_,norme_L2 = solver.corr_add(i,u_theta)
                tab_err_Corr[i,j] = norme_L2
            
        col_names = [("Corr",str(tab_nb_vert_Corr[i]),tab_h_Corr[i]) for i in range(len(tab_nb_vert_Corr))]
        mi = pd.MultiIndex.from_tuples(col_names, names=["method","n_vert","h"])
        df_Corr = pd.DataFrame(tab_err_Corr,columns=mi)
        df_Corr.to_csv(csv_file)
        
        df_Corr = pd.DataFrame(tab_err_Corr,columns=mi)
        df_Corr.to_csv(csv_file)

    return df_Corr, tab_nb_vert_Corr, tab_h_Corr, tab_err_Corr

def compute_error_Corr_all(n_params,problem,high_degree,u_theta,error_degree=4,new_run=False,result_dir="./"):
    for d in [1, 2, 3]:
        _, _, _, _ = compute_error_Corr_deg(n_params,problem,d,high_degree,u_theta,error_degree=error_degree,new_run=new_run,result_dir=result_dir)    
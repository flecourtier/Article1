import pandas as pd
import os
import numpy as np
from testcases.utils import get_random_params
from modfenics.error_estimations.utils import get_solver_type

def read_csv_Mult(csv_file):
    df_Mult = pd.read_csv(csv_file)
    tab_nb_vert_Mult = df_Mult.values[0,1:]
    tab_h_Mult = df_Mult.values[1,1:]
    tab_err_Mult = df_Mult.values[2:,1:]
    return df_Mult,tab_nb_vert_Mult,tab_h_Mult,tab_err_Mult

def compute_error_Mult_deg_M(n_params,problem,degree,high_degree,u_theta,M=0.0,error_degree=4,new_run=False,result_dir="./",impose_bc=True):
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
    
    csv_file = result_dir+f'Mult_case{testcase}_v{version}_degree{degree}_M{M}'
    if not impose_bc:
        csv_file += '_weak'
    csv_file += '.csv'
    if not new_run and os.path.exists(csv_file):
        print(f"## Read csv file {csv_file}")
        df_Mult, tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult = read_csv_Mult(csv_file)
    else:
        print(f"## Run gains with Mult for degree={degree}")
        tab_nb_vert_Mult = [20,40]
        tab_h_Mult = []
        tab_err_Mult = np.zeros((n_params,len(tab_nb_vert_Mult)))
        
        solver = solver_type(params=params, problem=problem, degree=degree, error_degree=error_degree, high_degree=high_degree, save_uref=save_uref)
        for (j,nb_vert) in enumerate(tab_nb_vert_Mult):
            print(f"nb_vert={nb_vert}")
            solver.set_meshsize(nb_cell=nb_vert-1)            
            tab_h_Mult.append(np.round(solver.h,3))
            
            for i in range(n_params):
                print(i,end=" ")
                _,_,norme_L2 = solver.corr_mult(i,u_theta,M=M,impose_bc=impose_bc)
                tab_err_Mult[i,j] = norme_L2
            
        col_names = [("Mult",str(tab_nb_vert_Mult[i]),tab_h_Mult[i]) for i in range(len(tab_nb_vert_Mult))]
        mi = pd.MultiIndex.from_tuples(col_names, names=["method","n_vert","h"])
        df_Mult = pd.DataFrame(tab_err_Mult,columns=mi)
        df_Mult.to_csv(csv_file)
        
        df_Mult = pd.DataFrame(tab_err_Mult,columns=mi)
        df_Mult.to_csv(csv_file)

    return df_Mult, tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult

def compute_error_Mult_deg_allM(n_params,problem,degree,high_degree,u_theta,tab_M,error_degree=4,new_run=False,result_dir="./"):
    for M in tab_M:
        _, _, _, _ = compute_error_Mult_deg_M(n_params,problem,degree,high_degree,u_theta,M=M,error_degree=error_degree,new_run=new_run,result_dir=result_dir)

def compute_error_Mult_alldeg_allM(n_params,problem,high_degree,u_theta,tab_M,error_degree=4,new_run=False,result_dir="./"):
    for d in [1, 2, 3]:
        compute_error_Mult_deg_allM(n_params,problem,d,high_degree,u_theta,tab_M,error_degree=error_degree,new_run=new_run,result_dir=result_dir)    
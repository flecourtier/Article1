import pandas as pd
import os
from modfenics.utils import get_param,compute_slope
from modfenics.error_estimations.utils import get_solver_type
from modfenics.error_estimations.fem import read_csv as read_csv_FEM
import matplotlib.pyplot as plt

def read_csv_Corr(csv_file):
    df_Corr = pd.read_csv(csv_file)
    tab_nb_vert_Corr = list(df_Corr['nb_vert'].values)
    tab_h_Corr = list(df_Corr['h'].values)
    tab_err_Corr = list(df_Corr['err'].values)
    
    return df_Corr,tab_nb_vert_Corr, tab_h_Corr, tab_err_Corr

def compute_error_estimations_Corr_deg(param_num,problem,degree,high_degree,u_theta,error_degree=4,new_run=False,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = [get_param(param_num,parameter_domain)]
    solver_type = get_solver_type(testcase,version)

    csv_file = result_dir+f'Corr_case{testcase}_v{version}_param{param_num}_degree{degree}.csv'
    if not new_run and os.path.exists(csv_file):
        print(f"## Read csv file {csv_file}")
        df_Corr,tab_nb_vert_Corr, tab_h_Corr, tab_err_Corr = read_csv_Corr(csv_file)
    else:
        print(f"## Run error estimation with Corr (add) for degree={degree}")
        tab_nb_vert_Corr = [2**i for i in range(4,9)]
        tab_h_Corr = []
        tab_err_Corr = []

        solver = solver_type(params=params, problem=problem, degree=degree, error_degree=error_degree, high_degree=high_degree)
        for nb_vert in tab_nb_vert_Corr:
            solver.set_meshsize(nb_cell=nb_vert-1)
            tab_h_Corr.append(solver.h)
            _,_,norme_L2 = solver.corr_add(0,u_theta)            
            print(f"nb_vert={nb_vert}, norme_L2={norme_L2}")
            tab_err_Corr.append(norme_L2)
            
        df_Corr = pd.DataFrame({'nb_vert': tab_nb_vert_Corr, 'h': tab_h_Corr, 'err': tab_err_Corr})
        df_Corr.to_csv(csv_file, index=False)
            
    return df_Corr,tab_nb_vert_Corr, tab_h_Corr, tab_err_Corr

def compute_error_estimations_Corr_all(param_num,problem,high_degree,u_theta,error_degree=4,new_run=False,result_dir="./",plot_cvg=False):
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = [get_param(param_num,parameter_domain)]
    
    if plot_cvg:
        plt.figure(figsize=(5, 5))

    dict = {}
    for d in [1, 2, 3]:
        df_Corr, tab_nb_vert_Corr, _, tab_err_Corr = compute_error_estimations_Corr_deg(param_num,problem,d,high_degree,u_theta,error_degree=error_degree,new_run=new_run,result_dir=result_dir)
        
        # to plot
        if plot_cvg:
            plt.loglog(df_Corr['nb_vert'], df_Corr['err'], "+-", label='P'+str(d))
        
            for i in range(1,len(tab_nb_vert_Corr)):
                slope, vert_mid = compute_slope(i,tab_nb_vert_Corr,tab_err_Corr)
                plt.text(vert_mid[0]+1e-2 , vert_mid[1], str(slope), fontsize=12, ha='left', va='top')
            
        # to save
        if d == 1:
            dict['N'] = tab_nb_vert_Corr
        dict[f'P{d}'] = tab_err_Corr
    
    if plot_cvg:
        plt.xticks(df_Corr['nb_vert'], df_Corr['nb_vert'].round(3).astype(str))
        plt.xlabel('nb_vert')
        plt.ylabel('L2 norm')
        plt.title(f'Corr case{testcase} v{version} param{param_num} : {params[0]}')
        plt.legend()
        plt.savefig(result_dir+f'Corr_case{testcase}_v{version}_param{param_num}.png')
        plt.show()
    
    # to save 
    df_deg = pd.DataFrame(dict)

    csv_file_all = result_dir+f'Corr_case{testcase}_v{version}_param{param_num}.csv'
    df_deg.to_csv(csv_file_all, index=False)
    
def plot_Corr_vs_FEM(param_num,problem,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = [get_param(param_num,parameter_domain)]
    
    plt.figure(figsize=(5, 5))

    # plot FEM error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        csv_file = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}_degree{d}.csv' 
        df_FEM,_,_,_ = read_csv_FEM(csv_file)
        plt.loglog(df_FEM['nb_vert'], df_FEM['err'], "+-", label='FEM P'+str(d))

    # plot Corr error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        csv_file = result_dir+f'Corr_case{testcase}_v{version}_param{param_num}_degree{d}.csv'
        df_Corr,_,_,_ = read_csv_Corr(csv_file)
        plt.loglog(df_Corr['nb_vert'], df_Corr['err'], ".--", label='Corr P'+str(d))

    plt.xticks(df_Corr['nb_vert'], df_Corr['nb_vert'].round(3).astype(str), minor=False)
    plt.xlabel("N")
    plt.ylabel('L2 norm')
    plt.legend()
    plt.title(f'FEM + Corr case{testcase} v{version} param{param_num} : {params[0]}')
    plt.savefig(result_dir+f'FEM-Corr_case{testcase}_v{version}_param{param_num}.png')
    plt.show()
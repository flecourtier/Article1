import pandas as pd
import os
from testcases.utils import create_tree,select_param,compute_slope
from modfenics.error_estimations.utils import get_solver_type
import matplotlib.pyplot as plt

def read_csv(csv_file):
    df_FEM = pd.read_csv(csv_file)
    tab_nb_vert_FEM = list(df_FEM['nb_vert'].values)
    tab_h_FEM = list(df_FEM['h'].values)
    tab_err_FEM = list(df_FEM['err'].values)
    
    return df_FEM,tab_nb_vert_FEM, tab_h_FEM, tab_err_FEM

def compute_error_estimations_fem_deg(param_num,problem,degree,high_degree,error_degree=4,new_run=False,result_dir="./",save_result=False):
    dim = problem.dim
    testcase = problem.testcase
    version = problem.version
    params = [select_param(problem,param_num)]     
    solver_type = get_solver_type(dim,testcase,version)
    
    save_uref = None
    if not problem.ana_sol:
        savedir = result_dir + "u_ref/"
        create_tree(savedir)
        filename = savedir + f"u_ref_{param_num}.npy"
        save_uref = [filename]
        print(filename)
    
    csv_file = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}_degree{degree}.csv' 
    if not new_run and os.path.exists(csv_file):
        print(f"## Read csv file {csv_file}")
        df_FEM,tab_nb_vert_FEM, tab_h_FEM, tab_err_FEM = read_csv(csv_file)
    else:
        print(f"## Run error estimation with FEM for degree={degree}")
        tab_nb_vert_FEM = [2**i for i in range(4,9)]
        tab_h_FEM = []
        tab_err_FEM = []
        solver = solver_type(params=params, problem=problem, degree=degree, error_degree=error_degree, high_degree=high_degree, save_uref=save_uref)
        for nb_vert in tab_nb_vert_FEM:
            solver.set_meshsize(nb_cell=nb_vert-1)
            tab_h_FEM.append(solver.h)
            fig_filename = None
            if save_result:
                result_dir_fig = result_dir + "FEM_plot/"
                create_tree(result_dir_fig)
                fig_filename = result_dir_fig + f'FEM_plot_case{testcase}_v{version}_param{param_num}_degree{degree}_N{nb_vert}.png'     
            _,norme_L2 = solver.fem(0,plot_result=False,filename=fig_filename)
            print(f"nb_vert={nb_vert}, norme_L2={norme_L2}")
            tab_err_FEM.append(norme_L2)
            
        df_FEM = pd.DataFrame({'nb_vert': tab_nb_vert_FEM, 'h': tab_h_FEM, 'err': tab_err_FEM})
        df_FEM.to_csv(csv_file, index=False)
        print(csv_file)
            
    return df_FEM,tab_nb_vert_FEM, tab_h_FEM, tab_err_FEM

def compute_error_estimations_fem_all(param_num,problem,high_degree,error_degree=4,new_run=False,result_dir="./",plot_cvg=False):
    testcase = problem.testcase
    version = problem.version
    params = [select_param(problem,param_num)]
    
    if plot_cvg:
        plt.figure(figsize=(5, 5))

    dict = {}
    for d in [1, 2, 3]:
        df_FEM, tab_nb_vert_FEM, _, tab_err_FEM = compute_error_estimations_fem_deg(param_num,problem,d,high_degree,error_degree=error_degree,new_run=new_run,result_dir=result_dir)
        
        # to plot
        if plot_cvg:
            plt.loglog(df_FEM['nb_vert'], df_FEM['err'], "+-", label='P'+str(d))
        
            for i in range(1,len(tab_nb_vert_FEM)):
                slope, vert_mid = compute_slope(i,tab_nb_vert_FEM,tab_err_FEM)
                plt.text(vert_mid[0]+1e-2 , vert_mid[1], str(slope), fontsize=12, ha='left', va='top')
            
        # to save
        if d == 1:
            dict['N'] = tab_nb_vert_FEM
        dict[f'P{d}'] = tab_err_FEM
    
    if plot_cvg:
        plt.xticks(df_FEM['nb_vert'], df_FEM['nb_vert'].round(3).astype(str))
        plt.xlabel('nb_vert')
        plt.ylabel('L2 norm')
        plt.title(f'FEM case{testcase} v{version} param{param_num} : {params[0]}')
        plt.legend()
        plt.savefig(result_dir+f'FEM_case{testcase}_v{version}_param{param_num}.png')
        plt.show()
    
    # to save 
    df_deg = pd.DataFrame(dict)

    csv_file_all = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}.csv'
    df_deg.to_csv(csv_file_all, index=False)
    
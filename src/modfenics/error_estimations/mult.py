import pandas as pd
import os
from modfenics.utils import get_param,compute_slope
from testcases.utils import create_tree
from modfenics.error_estimations.utils import get_solver_type
from modfenics.error_estimations.fem import read_csv as read_csv_FEM
import matplotlib.pyplot as plt

def read_csv_Mult(csv_file):
    df_Mult = pd.read_csv(csv_file)
    tab_nb_vert_Mult = list(df_Mult['nb_vert'].values)
    tab_h_Mult = list(df_Mult['h'].values)
    tab_err_Mult = list(df_Mult['err'].values)
    
    return df_Mult,tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult

def compute_error_estimations_Mult_deg(param_num,problem,degree,high_degree,u_theta,M=0.0,error_degree=4,new_run=False,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = [get_param(param_num,parameter_domain)]
    solver_type = get_solver_type(testcase,version)
    
    save_uref = None
    if not problem.ana_sol:
        savedir = result_dir + "u_ref/"
        create_tree(savedir)
        filename = savedir + f"u_ref_{param_num}.npy"
        save_uref = [filename]
        print(filename)

    csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{degree}_M{M}.csv'
    if not new_run and os.path.exists(csv_file):
        print(f"## Read csv file {csv_file}")
        df_Mult,tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult = read_csv_Mult(csv_file)
    else:
        print(f"## Run error estimation with Corr (mult) for degree={degree}")
        tab_nb_vert_Mult = [2**i for i in range(4,9)]
        tab_h_Mult = []
        tab_err_Mult = []

        solver = solver_type(params=params, problem=problem, degree=degree, error_degree=error_degree, high_degree=high_degree, save_uref=save_uref)
        for nb_vert in tab_nb_vert_Mult:
            solver.set_meshsize(nb_cell=nb_vert-1)
            tab_h_Mult.append(solver.h)
            _,_,norme_L2 = solver.corr_mult(0,u_theta,M=M)            
            print(f"nb_vert={nb_vert}, norme_L2={norme_L2}")
            tab_err_Mult.append(norme_L2)
            
        df_Mult = pd.DataFrame({'nb_vert': tab_nb_vert_Mult, 'h': tab_h_Mult, 'err': tab_err_Mult})
        df_Mult.to_csv(csv_file, index=False)
            
    return df_Mult,tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult

def compute_error_estimations_Mult_all(param_num,problem,high_degree,u_theta,M=0.0,error_degree=4,new_run=False,result_dir="./",plot_cvg=False):
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = [get_param(param_num,parameter_domain)]
    
    if plot_cvg:
        plt.figure(figsize=(5, 5))

    dict = {}
    for d in [1, 2, 3]:
        df_Mult, tab_nb_vert_Mult, _, tab_err_Mult = compute_error_estimations_Mult_deg(param_num,problem,d,high_degree,u_theta,M=M,error_degree=error_degree,new_run=new_run,result_dir=result_dir)
        
        # to plot
        if plot_cvg:
            plt.loglog(df_Mult['nb_vert'], df_Mult['err'], "+-", label='P'+str(d))
        
            for i in range(1,len(tab_nb_vert_Mult)):
                slope, vert_mid = compute_slope(i,tab_nb_vert_Mult,tab_err_Mult)
                plt.text(vert_mid[0]+1e-2 , vert_mid[1], str(slope), fontsize=12, ha='left', va='top')
            
        # to save
        if d == 1:
            dict['N'] = tab_nb_vert_Mult
        dict[f'P{d}'] = tab_err_Mult
    
    if plot_cvg:
        plt.xticks(df_Mult['nb_vert'], df_Mult['nb_vert'].round(3).astype(str))
        plt.xlabel('nb_vert')
        plt.ylabel('L2 norm')
        plt.title(f'Mult case{testcase} v{version} param{param_num} : {params[0]} (M={M})')
        plt.legend()
        plt.savefig(result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_M{M}.png')
        plt.show()
    
    # to save 
    df_deg = pd.DataFrame(dict)

    csv_file_all = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}.csv'
    df_deg.to_csv(csv_file_all, index=False)
    
def plot_Mult_vs_FEM(param_num,problem,M=0.0,result_dir="./"):
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

    # plot Mult error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{d}_M{M}.csv'
        df_Mult,_,_,_ = read_csv_Mult(csv_file)
        plt.loglog(df_Mult['nb_vert'], df_Mult['err'], ".--", label='Mult P'+str(d))

    plt.xticks(df_Mult['nb_vert'], df_Mult['nb_vert'].round(3).astype(str), minor=False)
    plt.xlabel("N")
    plt.ylabel('L2 norm')
    plt.legend()
    plt.title(f'FEM + Mult case{testcase} v{version} param{param_num} : {params[0]} (M={M})')
    plt.savefig(result_dir+f'FEM-Mult_case{testcase}_v{version}_param{param_num}_M{M}.png')
    plt.show()
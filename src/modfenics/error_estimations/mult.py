import pandas as pd
import os
from testcases.utils import create_tree,select_param,compute_slope
from modfenics.error_estimations.utils import get_solver_type
from modfenics.error_estimations.fem import read_csv as read_csv_FEM
from modfenics.error_estimations.add import read_csv_Corr as read_csv_Corr
import matplotlib.pyplot as plt

def read_csv_Mult(csv_file):
    df_Mult = pd.read_csv(csv_file)
    tab_nb_vert_Mult = list(df_Mult['nb_vert'].values)
    tab_h_Mult = list(df_Mult['h'].values)
    tab_err_Mult = list(df_Mult['err'].values)
    
    return df_Mult,tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult

def compute_error_estimations_Mult_deg(param_num,problem,degree,high_degree,u_theta,M=0.0,error_degree=4,new_run=False,result_dir="./",impose_bc=True,save_result=False):
    # impose_bc: strong imposition of boundary conditions
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

    csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{degree}_M{M}'
    if not impose_bc:
        csv_file += '_weak'
    csv_file += '.csv'
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
            fig_filename = None
            if save_result:
                result_dir_fig = result_dir + "Mult_plot"
                if not impose_bc:
                    result_dir_fig += "_weak"
                result_dir_fig += "/"
                create_tree(result_dir_fig)
                fig_filename = result_dir_fig + f'Mult_plot_case{testcase}_v{version}_param{param_num}_degree{degree}_N{nb_vert}_M{M}'
                if not impose_bc:
                    fig_filename += '_weak'
                fig_filename += '.png'
            _,_,norme_L2 = solver.corr_mult(0,u_theta,M=M,impose_bc=impose_bc,plot_result=False,filename=fig_filename)           
            print(f"nb_vert={nb_vert}, norme_L2={norme_L2}")
            tab_err_Mult.append(norme_L2)
            
        df_Mult = pd.DataFrame({'nb_vert': tab_nb_vert_Mult, 'h': tab_h_Mult, 'err': tab_err_Mult})
        df_Mult.to_csv(csv_file, index=False)
            
    return df_Mult,tab_nb_vert_Mult, tab_h_Mult, tab_err_Mult

def compute_error_estimations_Mult_all(param_num,problem,high_degree,u_theta,M=0.0,error_degree=4,new_run=False,result_dir="./",plot_cvg=False):
    testcase = problem.testcase
    version = problem.version
    params = [select_param(problem,param_num)]
    
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
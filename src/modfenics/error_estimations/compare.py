from testcases.utils import select_param
from modfenics.error_estimations.fem import read_csv as read_csv_FEM
from modfenics.error_estimations.add import read_csv_Corr as read_csv_Add
from modfenics.error_estimations.mult import read_csv_Mult
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi
import pandas as pd

def plot_Corr_vs_FEM(param_num,problem,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    params = [select_param(problem,param_num)]
    
    plt.figure(figsize=(5, 5))

    # plot FEM error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        try:
            csv_file = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}_degree{d}.csv' 
            df_FEM,_,_,_ = read_csv_FEM(csv_file)
            plt.loglog(df_FEM['nb_vert'], df_FEM['err'], "+-", label='FEM P'+str(d))
        except:
            print(f'FEM P{d} not found')

    # plot Corr error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        try:
            csv_file = result_dir+f'Corr_case{testcase}_v{version}_param{param_num}_degree{d}.csv'
            df_Corr,_,_,_ = read_csv_Add(csv_file)
            plt.loglog(df_Corr['nb_vert'], df_Corr['err'], ".--", label='Corr P'+str(d))
        except:
            print(f'Corr P{d} not found')
            
    plt.xticks(df_Corr['nb_vert'], df_Corr['nb_vert'].round(3).astype(str), minor=False)
    plt.xlabel("N")
    plt.ylabel('L2 norm')
    plt.legend()
    plt.title(f'FEM + Corr case{testcase} v{version} param{param_num} : {params[0]}')
    plt.savefig(result_dir+f'FEM-Corr_case{testcase}_v{version}_param{param_num}.png')
    plt.show()
    
def plot_Mult_vs_FEM(param_num,problem,M=0.0,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    params = [select_param(problem,param_num)]
    
    plt.figure(figsize=(5, 5))

    # plot FEM error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        try:
            csv_file = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}_degree{d}.csv' 
            df_FEM,_,_,_ = read_csv_FEM(csv_file)
            plt.loglog(df_FEM['nb_vert'], df_FEM['err'], "+-", label='FEM P'+str(d))
        except:
            print(f'FEM P{d} not found')
            
    # plot Mult error (L2 norm) as a function of h
    for d in [1, 2, 3]:
        try:
            csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{d}_M{M}.csv'
            df_Mult,_,_,_ = read_csv_Mult(csv_file)
            plt.loglog(df_Mult['nb_vert'], df_Mult['err'], ".--", label='Mult P'+str(d))
        except:
            print(f'Mult P{d} not found')

    plt.xticks(df_Mult['nb_vert'], df_Mult['nb_vert'].round(3).astype(str), minor=False)
    plt.xlabel("N")
    plt.ylabel('L2 norm')
    plt.legend()
    plt.title(f'FEM + Mult case{testcase} v{version} param{param_num} : {params[0]} (M={M})')
    plt.savefig(result_dir+f'FEM-Mult_case{testcase}_v{version}_param{param_num}_M{M}.png')
    plt.show()
    
def plot_Mult_vs_Add_vs_FEM(param_num,problem,degree,tab_M,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    params = [select_param(problem,param_num)]
    
    plt.figure(figsize=(5, 5))

    # plot FEM error (L2 norm) as a function of h
    try:
        csv_file = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}_degree{degree}.csv' 
        df_FEM,_,_,_ = read_csv_FEM(csv_file)
        plt.loglog(df_FEM['nb_vert'], df_FEM['err'], "+-", label='FEM P'+str(degree))
    except:
        print(f'FEM P{degree} not found')
    
    try:    
        csv_file = result_dir+f'Corr_case{testcase}_v{version}_param{param_num}_degree{degree}.csv'
        df_Add,_,_,_ = read_csv_Add(csv_file)
        plt.loglog(df_Add['nb_vert'], df_Add['err'], ".--", label='Add P'+str(degree))
    except:
        print(f'Add P{degree} not found')
        
    # plot Mult error (L2 norm) as a function of h
    for M in tab_M:
        try:
            csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{degree}_M{M}.csv'
            df_Mult,_,_,_ = read_csv_Mult(csv_file)
            plt.loglog(df_Mult['nb_vert'], df_Mult['err'], ".--", label='Mult_s P'+str(degree)+' M = '+str(M))
        except:
            print(f'Mult strong P{degree} M{M} not found')
        
        try:
            csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{degree}_M{M}_weak.csv'
            df_Mult,_,_,_ = read_csv_Mult(csv_file)
            plt.loglog(df_Mult['nb_vert'], df_Mult['err'], ".--", label='Mult_w P'+str(degree)+' M = '+str(M))
        except:
            print(f'Mult weak P{degree} M{M} not found')
        
    plt.xticks(df_Mult['nb_vert'], df_Mult['nb_vert'].round(3).astype(str), minor=False)
    plt.xlabel("N")
    plt.ylabel('L2 norm')
    plt.legend()
    plt.title(f'FEM + Add + Mult case{testcase} v{version} param{param_num} deg{degree} : {params[0]}')
    plt.savefig(result_dir+f'FEM-Add-Mult_case{testcase}_v{version}_param{param_num}.png')
    plt.show()
    
def save_tab(param_num,problem,degree,tab_M=None,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    # params = [select_param(problem,param_num)]
    tab_vals = []
    iterables = []
    
    try:
        csv_file = result_dir+f'FEM_case{testcase}_v{version}_param{param_num}_degree{degree}.csv' 
        _,tab_nb_vert_FEM,_,tab_err_FEM = read_csv_FEM(csv_file)
        tab_err_FEM = np.array(tab_err_FEM)
        tab_vals.append(tab_err_FEM)
        iterables.append(("FEM","error"))
    except:
        print(f'FEM P{degree} not found')
    
    try:
        csv_file = result_dir+f'Corr_case{testcase}_v{version}_param{param_num}_degree{degree}.csv'
        _,_,_,tab_err_Add = read_csv_Add(csv_file)
        tab_err_Add = np.array(tab_err_Add)
        facteurs_Add = tab_err_FEM/tab_err_Add
        
        tab_vals.append(tab_err_Add)
        tab_vals.append(facteurs_Add)
        iterables.append(("Corr","error"))
        iterables.append(("Corr","facteurs"))
    except:
        print(f'Corr P{degree} not found')
        
    # plot Mult error (L2 norm) as a function of h
    if tab_M is not None:
        for M in tab_M:
            try:
                csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{degree}_M{M}.csv'
                _,_,_,tab_err_Mult = read_csv_Mult(csv_file)
                tab_err_Mult = np.array(tab_err_Mult)
                facteurs_Mult = tab_err_FEM/tab_err_Mult
                tab_vals.append(tab_err_Mult)
                tab_vals.append(facteurs_Mult)
                iterables.append(("Mult"+str(M),"error"))
                iterables.append(("Mult"+str(M),"facteurs"))
            except:
                print(f'Mult strong P{degree} M{M} not found')
            
            try:
                csv_file = result_dir+f'Mult_case{testcase}_v{version}_param{param_num}_degree{degree}_M{M}_weak.csv'
                _,_,_,tab_err_Mult = read_csv_Mult(csv_file)
                tab_err_Mult = np.array(tab_err_Mult)
                facteurs_Mult = tab_err_FEM/tab_err_Mult
                tab_vals.append(tab_err_Mult)
                tab_vals.append(facteurs_Mult)
                iterables.append(("Mult"+str(M)+"w","error"))
                iterables.append(("Mult"+str(M)+"w","facteurs"))
            except:
                print(f'Mult weak P{degree} M{M} not found')

    index = pd.MultiIndex.from_tuples(iterables, names=["method", "type"])
    df = pd.DataFrame(tab_vals, index=index, columns=tab_nb_vert_FEM).T

    # Appliquer des formats spécifiques en fonction du type
    def custom_formatting(df):
        # Appliquer un format spécifique pour les erreurs (notation scientifique)
        error_cols = df.columns[df.columns.get_level_values('type') == 'error']
        df[error_cols] = df[error_cols].applymap(lambda x: f'{x:.2e}')
        
        # Arrondir les facteurs à l'entier le plus proche
        factor_cols = df.columns[df.columns.get_level_values('type') == 'facteurs']
        df[factor_cols] = df[factor_cols].applymap(lambda x: f'{round(x,2)}')

        return df

    # Appliquer la fonction de mise en forme
    formatted_df = custom_formatting(df)

    # Sauvegarder le DataFrame formaté au format CSV
    formatted_df.to_csv(result_dir+f'Tab_case{testcase}_v{version}_param{param_num}_degree{degree}.csv')
    # Et au format PNG

    dfi.export(formatted_df, result_dir+f'Tab_case{testcase}_v{version}_param{param_num}_degree{degree}.png', dpi=300)
import pandas as pd
import numpy as np
import dataframe_image as dfi

from modfenics.gains.fem import read_csv as read_csv_FEM
from modfenics.gains.pinns import read_csv_PINNs
from modfenics.gains.add import read_csv_Corr as read_csv_Add
from testcases.utils import get_random_params

def create_dataframes_deg(n_params,problem,degree,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    parameter_domain = problem.parameter_domain
    params = get_random_params(n_params,parameter_domain)
    params_str = np.array([f"{params[i][0].round(2)},{params[i][1].round(2)}" for i in range(n_params)])
    
    try:
        csv_file = result_dir+f'FEM_errors_case{testcase}_v{version}_degree{degree}.csv'
        _,tab_nb_vert_FEM,tab_h_FEM,tab_err_FEM = read_csv_FEM(csv_file)
    except:
        raise FileNotFoundError(f'FEM P{degree} not found')        
        
    try:
        csv_file = result_dir+f'PINNs_errors_case{testcase}_v{version}_degree{degree}.csv'
        _,tab_nb_vert_PINNs,tab_h_PINNs,tab_err_PINNs = read_csv_PINNs(csv_file)
    except:
        raise FileNotFoundError(f'PINNs P{degree} not found')
                
    try:
        csv_file = result_dir+f'Corr_errors_case{testcase}_v{version}_degree{degree}.csv'
        _,tab_nb_vert_Corr,tab_h_Corr,tab_err_Corr = read_csv_Add(csv_file)
    except:
        raise FileNotFoundError(f'Corr P{degree} not found')

    # check if number of vertices are the same
    assert tab_nb_vert_FEM.all() == tab_nb_vert_PINNs.all() and tab_nb_vert_FEM.all() == tab_nb_vert_Corr.all(), "Different number of vertices, cannot compute gains between methods"

    tab_nb_vert = tab_nb_vert_FEM
    tab_h = tab_h_FEM
    row_names = [str(i) + " : " + params_str[i] for i in range(n_params)]

    # create dataframe for errors
    col_names = [("FEM",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(len(tab_nb_vert))] + \
        [("PINNs",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(len(tab_nb_vert))] + \
        [("Corr",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(len(tab_nb_vert))]
    mi = pd.MultiIndex.from_tuples(col_names, names=["method","n_vert","h"])
    df_errors = pd.DataFrame(columns=mi,index=row_names)
    
    for i in range(n_params):
        for j in range(len(tab_nb_vert_FEM)):
            df_errors.loc[row_names[i],col_names[j]] = tab_err_FEM[i,j]
            j2=j+1
        for j in range(len(tab_nb_vert_PINNs)):
            df_errors.loc[row_names[i],col_names[j2+j]] = tab_err_PINNs[i,j]
            j3 = j2+j+1
        for j in range(len(tab_nb_vert_Corr)):
            df_errors.loc[row_names[i],col_names[j3+j]] = tab_err_Corr[i,j]
    
    save_file = result_dir+f'df_errors_case{testcase}_v{version}_degree{degree}.csv'
    df_errors.to_csv(save_file)
    
    # create dataframe for gains
    gains_PINNs_Corr = df_errors["PINNs"] / df_errors["Corr"]
    gains_FEM_PINNs = df_errors["FEM"] / df_errors["PINNs"]
    gains_FEM_Corr = df_errors["FEM"] / df_errors["Corr"]
    
    col_names = [("FEM/PINNs",str(tab_nb_vert_FEM[i]),str(tab_h[i])) for i in range(len(tab_nb_vert))] + \
        [("PINNs/Corr",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(len(tab_nb_vert))] + \
        [("FEM/Corr",str(tab_nb_vert[i]),str(tab_h[i])) for i in range(len(tab_nb_vert))]
    mi = pd.MultiIndex.from_tuples(col_names, names=["facteurs","n_vert","h"])
    df_gains = pd.DataFrame(columns=mi,index=row_names)

    # fill dataframes
    for i in range(n_params):
        for j in range(len(tab_nb_vert_FEM)):
            df_gains.loc[row_names[i],col_names[j]] = gains_FEM_PINNs.to_numpy()[i,j]
            j2=j+1
        for j in range(len(tab_nb_vert_PINNs)):
            df_gains.loc[row_names[i],col_names[j2+j]] = gains_PINNs_Corr.to_numpy()[i,j]
            j3 = j2+j+1
        for j in range(len(tab_nb_vert_Corr)):
            df_gains.loc[row_names[i],col_names[j3+j]] = gains_FEM_Corr.to_numpy()[i,j]
    
    save_file = result_dir+f'df_gains_case{testcase}_v{version}_degree{degree}.csv'            
    df_gains.to_csv(save_file)
        
    return df_errors,df_gains
    
def create_dataframes_all(n_params,problem,result_dir="./"):
    for d in [1,2,3]:
        _,_ = create_dataframes_deg(n_params,problem,d,result_dir=result_dir)

def save_stats_deg(n_params,problem,degree,result_dir="./"):
    testcase = problem.testcase
    version = problem.version
    
    _,df_gains = create_dataframes_deg(n_params,problem,degree,result_dir=result_dir)
    
    try:
        csv_file = result_dir+f'FEM_errors_case{testcase}_v{version}_degree{degree}.csv'
        _,tab_nb_vert,_,_ = read_csv_FEM(csv_file)
    except:
        raise FileNotFoundError(f'FEM P{degree} not found')        
    
    def get_df_facteurs_n_vert(i):
        n_vert = tab_nb_vert[i]
        
        # on crée une dataframe contenant les facteurs pour chaque méthode avec n_vert=n_vert
        df_facteurs_n_vert = df_gains[[col for col in df_gains.columns if col[1] == str(n_vert)]]
        # on supprime la première colonne 
        df_facteurs_n_vert = df_facteurs_n_vert.drop(columns=[df_facteurs_n_vert.columns[0]])
        
        # on change les noms des colonnes
        df_facteurs_n_vert.columns = [col[0] for col in df_facteurs_n_vert.columns]
        
        return df_facteurs_n_vert

    def get_values(df_facteurs_n_vert):
        df_min = df_facteurs_n_vert.min(axis=0)
        df_max = df_facteurs_n_vert.max(axis=0)
        df_mean = df_facteurs_n_vert.mean(axis=0)
        df_std = df_facteurs_n_vert.std(axis=0)
        
        return [df_min["PINNs/Corr"],df_max["PINNs/Corr"],df_mean["PINNs/Corr"],df_std["PINNs/Corr"]], \
                [df_min["FEM/Corr"],df_max["FEM/Corr"],df_mean["FEM/Corr"],df_std["FEM/Corr"]]

    tab_gains_Add_on_PINNs = []
    tab_gains_Add_on_FEM = []

    for i in range(len(tab_nb_vert)):
        df_facteurs_n_vert = get_df_facteurs_n_vert(i)
        gains_Add_on_PINNs,gains_Add_on_FEM = get_values(df_facteurs_n_vert)
        tab_gains_Add_on_PINNs.append(gains_Add_on_PINNs)
        tab_gains_Add_on_FEM.append(gains_Add_on_FEM)
        
    tab_gains_Add_on_PINNs = np.array(tab_gains_Add_on_PINNs)
    tab_gains_Add_on_FEM = np.array(tab_gains_Add_on_FEM)

    columns= ["min_PINNs","max_PINNs","mean_PINNs","std_PINNs","min_FEM","max_FEM","mean_FEM","std_FEM"]

    df_stats_Add = pd.DataFrame(np.concatenate([tab_gains_Add_on_PINNs,tab_gains_Add_on_FEM],axis=1),columns=columns,index=tab_nb_vert)
    df_stats_Add.index.name = "N"
        
    df_stats = pd.concat([df_stats_Add],keys=["Add"],names=["method"])
    df_stats
    
    result_file = result_dir+f'Tab_stats_case{testcase}_v{version}_degree{degree}'

    df_stats.to_csv(result_file+'.csv')
    df_stats_round = df_stats.round(1)
    # df_stats_round = df_stats_round.astype(int)

    dfi.export(df_stats_round,result_file+".png",dpi=1000)
    
    return df_stats
    
def save_stats_all(n_params,problem,result_dir="./"):
    for d in [1,2,3]:
        _ = save_stats_deg(n_params,problem,d,result_dir=result_dir)
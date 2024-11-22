import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi
import os

from modfenics.error_estimations.error_estimations import ErrorEstimations

class CompareMethods:
    def __init__(self,error_estimations:ErrorEstimations):
        self.ee = error_estimations
        
    #########
    # Plots #
    #########

    def plot_method_vs_FEM_alldeg(self,method,**kwargs): 
        assert method in ["Corr","Mult"], f"method={method} can't be compared with FEM"  
        if method == "Mult":
            assert 'M' in kwargs and 'impose_bc' in kwargs, f"M and impose_bc are required for {method}"
            M = kwargs.get('M')
            impose_bc = kwargs.get('impose_bc')
        else:
            assert not 'M' in kwargs and not 'impose_bc' in kwargs, f"M and impose_bc are not required for {method}"

        plt.figure(figsize=(5, 5))

        # plot FEM vs method error (L2 norm) as a function of h for all degrees
        for degree in self.ee.tab_degree:
            try: 
                csv_file = self.ee.results_dir+f'FEM_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.csv'
                print(csv_file)
                df_FEM,_,_ = self.ee.read_csv(csv_file)
                plt.loglog(self.ee.tab_nb_vert, df_FEM['err'], "+-", label=f'FEM P{degree}')
            except:
                print(f'FEM P{degree} not found')

        for degree in self.ee.tab_degree:
            try:
                csv_file = self.ee.results_dir+f'{method}_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}'
                if method == "Mult":
                    csv_file += f"_M{M}"
                    csv_file += '_weak' if not impose_bc else ''
                csv_file += ".csv"
                df_method,_,_ = self.ee.read_csv(csv_file)
                plt.loglog(self.ee.tab_nb_vert, df_method['err'], ".--", label=f'{method} P{degree}')
            except:
                if method != "Mult":
                    print(f'{method} P{degree} not found')
                else:
                    type = "(weak) " if not impose_bc else ""
                    print(f'{method} {type}P{degree} M{M} not found (M={M})')
                
        plt.xticks(self.ee.tab_nb_vert, np.array(self.ee.tab_nb_vert).round(3).astype(str), minor=False)
        plt.xlabel("N")
        plt.ylabel('L2 norm')
        plt.legend()
        if method != "Mult":
            title = f'FEM + {method} case{self.ee.testcase} v{self.ee.version} param{self.ee.param_num} : {self.ee.params[0]}'
            fig_filename = self.ee.results_dir+f'FEM-{method}_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}.png'
        else:
            type = "_weak" if not impose_bc else ""
            title = f'FEM + {method} {type}case{self.ee.testcase} v{self.ee.version} param{self.ee.param_num} : {self.ee.params[0]} (M={M})'
            fig_filename = self.ee.results_dir+f'FEM-{method}_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_M{M}{type}.png'
        plt.title(title)
        plt.savefig(fig_filename)
        plt.show()
        
    def plot_Corr_vs_FEM_alldeg(self):
        self.plot_method_vs_FEM_alldeg("Corr")
        
    def plot_Mult_vs_FEM_alldeg_M(self,M=0.0,impose_bc=True):
        self.plot_method_vs_FEM_alldeg("Mult",M=M,impose_bc=impose_bc)
        
    def plot_Mult_vs_FEM_alldeg_allM(self,tab_M,impose_bc=True):
        for M in tab_M:
            self.plot_Mult_vs_FEM_alldeg_M(M=M,impose_bc=impose_bc)

    def plot_Mult_vs_Add_vs_FEM_deg_allM(self,degree,tab_M):
        plt.figure(figsize=(5, 5))

        df_FEM = None
        # plot FEM error (L2 norm) as a function of h
        try:
            csv_file = self.ee.results_dir+f'FEM_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.csv' 
            df_FEM,_,_ = self.ee.read_csv(csv_file)
            plt.loglog(df_FEM['nb_vert'], df_FEM['err'], "+-", label='FEM P'+str(degree))
        except:
            print(f'FEM P{degree} not found')
        
        df_Add = None
        try:    
            csv_file = self.ee.results_dir+f'Corr_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.csv'
            df_Add,_,_ = self.ee.read_csv(csv_file)
            plt.loglog(df_Add['nb_vert'], df_Add['err'], ".--", label='Add P'+str(degree))
        except:
            print(f'Add P{degree} not found')
        
        df_Mult = None
        # plot Mult error (L2 norm) as a function of h
        for M in tab_M:
            try:
                csv_file = self.ee.results_dir+f'Mult_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}_M{M}.csv'
                print(csv_file)
                df_Mult,_,_ = self.ee.read_csv(csv_file)
                plt.loglog(df_Mult['nb_vert'], df_Mult['err'], ".--", label='Mult_s P'+str(degree)+' M = '+str(M))
            except:
                print(f'Mult strong P{degree} M{M} not found')
            
            try:
                csv_file = self.ee.results_dir+f'Mult_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}_M{M}_weak.csv'
                df_Mult,_,_ = self.ee.read_csv(csv_file)
                plt.loglog(df_Mult['nb_vert'], df_Mult['err'], ".--", label='Mult_w P'+str(degree)+' M = '+str(M))
            except:
                print(f'Mult weak P{degree} M{M} not found')
                
        # si une des dataframe existe
        if df_FEM is not None or df_Add is not None or df_Mult is not None:   
            plt.xticks(df_FEM['nb_vert'], df_FEM['nb_vert'].round(3).astype(str), minor=False)
            plt.xlabel("N")
            plt.ylabel('L2 norm')
            plt.legend()
            plt.title(f'FEM + Add + Mult case{self.ee.testcase} v{self.ee.version} param{self.ee.param_num} deg{degree} : {self.ee.params[0]}')
            plt.savefig(self.ee.results_dir+f'FEM-Add-Mult_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.png')
            plt.show()
        else:
            print(f'No data found for param{self.ee.param_num} deg{degree}')
            plt.close()
    
    def plot_Mult_vs_Add_vs_FEM_alldeg_allM(self,tab_M):
        for degree in self.ee.tab_degree:
            self.plot_Mult_vs_Add_vs_FEM_deg_allM(degree,tab_M)
        
    ##########
    # Tables #
    ##########
    
    def save_tab_deg_allM(self,degree,tab_M=None):
        tab_vals = []
        iterables = []
        
        try:
            csv_file = self.ee.results_dir+f'FEM_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.csv' 
            _,_,tab_err_FEM = self.ee.read_csv(csv_file)
            tab_err_FEM = np.array(tab_err_FEM)
            tab_vals.append(tab_err_FEM)
            iterables.append(("FEM","error"))
        except:
            print(f'FEM P{degree} not found')
        
        try:
            csv_file = self.ee.results_dir+f'Corr_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.csv'
            _,_,tab_err_Add = self.ee.read_csv(csv_file)
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
                    csv_file = self.ee.results_dir+f'Mult_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}_M{M}.csv'
                    _,_,tab_err_Mult = self.ee.read_csv(csv_file)
                    tab_err_Mult = np.array(tab_err_Mult)
                    facteurs_Mult = tab_err_FEM/tab_err_Mult
                    tab_vals.append(tab_err_Mult)
                    tab_vals.append(facteurs_Mult)
                    iterables.append(("Mult"+str(M),"error"))
                    iterables.append(("Mult"+str(M),"facteurs"))
                except:
                    print(f'Mult strong P{degree} M{M} not found')
                
                try:
                    csv_file = self.ee.results_dir+f'Mult_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}_M{M}_weak.csv'
                    _,_,tab_err_Mult = self.ee.read_csv(csv_file)
                    tab_err_Mult = np.array(tab_err_Mult)
                    facteurs_Mult = tab_err_FEM/tab_err_Mult
                    tab_vals.append(tab_err_Mult)
                    tab_vals.append(facteurs_Mult)
                    iterables.append(("Mult"+str(M)+"w","error"))
                    iterables.append(("Mult"+str(M)+"w","facteurs"))
                except:
                    print(f'Mult weak P{degree} M{M} not found')

        index = pd.MultiIndex.from_tuples(iterables, names=["method", "type"])
        df = pd.DataFrame(tab_vals, index=index, columns=self.ee.tab_nb_vert).T

        # Appliquer des formats spécifiques en fonction du type
        def custom_formatting(df):
            # Appliquer un format spécifique pour les erreurs (notation scientifique)
            error_cols = df.columns[df.columns.get_level_values('type') == 'error']
            df[error_cols] = df[error_cols].applymap(lambda x: f'{x:.2e}')
            
            # Arrondir les facteurs à l'entier le plus proche
            factor_cols = df.columns[df.columns.get_level_values('type') == 'facteurs']
            df[factor_cols] = df[factor_cols].applymap(lambda x: f'{round(x,2)}')

            return df
        
        # Si le DataFrame est vide, ne pas le sauvegarder
        if not df.empty:

            # Appliquer la fonction de mise en forme
            formatted_df = custom_formatting(df)
            
            # Sauvegarder le DataFrame formaté au format CSV et PNG
            filename = self.ee.results_dir+f'Tab_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}'
            formatted_df.to_csv(filename+'.csv')
            # table_conversion = "chrome"
            table_conversion = "matplotlib"
            dfi.export(formatted_df, filename+'.png', dpi=300, table_conversion=table_conversion)
        
    def save_tab_alldeg_allM(self,tab_M=None):
        for degree in self.ee.tab_degree:
            self.save_tab_deg_allM(degree,tab_M)
            
class CompareMethodsMeshSize(CompareMethods):
    def __get_index(self,tab,val):
        tab = np.array(tab)
        if val < tab[-1]:
            return len(tab)-1
        if val > tab[0]:
            return 1
        return np.where(tab < val)[0][0]
    
    def __linear_interpolation_on_x(self,tab,given_y):
        tab_x,tab_y = tab[:,0],tab[:,1]
        index = self.__get_index(tab_y,given_y)
        
        tab_x = np.log(np.array(tab_x))
        tab_y = np.log(np.array(tab_y)) 

        x = tab_x[index-1:index+1]
        y = tab_y[index-1:index+1]
        given_y = np.log(given_y)
        
        pente = (y[1]-y[0])/(x[1]-x[0])
        x_inter = x[0]+(given_y-y[0])/pente
        
        return np.exp(x_inter)
    
    def get_N_deg_M(self,method,given_precision,degree,M=None,impose_bc=True):
        assert method in ["FEM","Corr","Mult"], f"method={method} is not implemented"
        if method == "Mult":
            assert M is not None, f"M is required for {method}"
        
        try:
            csv_file = self.ee.results_dir+f'{method}_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}'
            if method == "Mult":
                csv_file += f"_M{M}"
                csv_file += '_weak' if not impose_bc else ''
            csv_file += ".csv"
        except:
            print(f'{method} P{degree} not found')
        
        
        _,_,tab_err = self.ee.read_csv(csv_file)
        tab_N = self.ee.tab_nb_vert
        tab = np.column_stack((tab_N,tab_err))
        
        return self.__linear_interpolation_on_x(tab,given_precision)

    def get_N_at_given_precision_deg_allM(self,given_precision,degree,tab_M=None):
        tab_methods = ["FEM","Corr"]
        if tab_M is not None:
            tab_methods.append("Mult")
        
        tab_N = {}
        for method in tab_methods:
            if method == "Mult":
                for M in tab_M:
                    impose_bc = True
                    N = self.get_N_deg_M(method,given_precision,degree,M,impose_bc)
                    tab_N[method+str(M)] = N
                    
                    impose_bc = False
                    N = self.get_N_deg_M(method,given_precision,degree,M,impose_bc)
                    tab_N[method+str(M)+"_weak"] = N
            else:
                N = self.get_N_deg_M(method,given_precision,degree)
                tab_N[method] = N
        
        return tab_N
    
    def save_tab_given_precisions_deg_allM(self,degree,tab_M=None,tab_given_precision = [1e-3,1e-4]):
        result_dir = self.ee.results_dir + "TabN/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        tab_N_prec = {}
        for given_precision in tab_given_precision:
            tab_N = self.get_N_at_given_precision_deg_allM(given_precision,degree,tab_M)
            for key in tab_N.keys():
                if not key in tab_N_prec:
                    tab_N_prec[key] = []
                tab_N_prec[key].append(tab_N[key])
            tab_methods = list(tab_N.keys())
        
        df = pd.DataFrame(tab_N_prec, index=tab_given_precision, columns=tab_methods)
        
        # apply scientific formats for the index
        df.index = df.index.map(lambda x: f'{x:.0e}')
        
        # Sauvegarder le DataFrame au format CSV et PNG
        filename = result_dir+f'TabN_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}'
        df.to_csv(filename+'.csv')
        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        dfi.export(df, filename+'.png', dpi=300, table_conversion=table_conversion)
        
        return df
    
    def save_tab_given_precisions_alldeg_allM(self,tab_M=None):
        result_dir = self.ee.results_dir + "TabN/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        df_all = {}
        for degree in self.ee.tab_degree:
            tab_given_precision = [10**-(degree+2),10**-(degree+3)]
            df_deg = self.save_tab_given_precisions_deg_allM(degree,tab_M,tab_given_precision)
            df_all[degree] = df_deg
            
        df = pd.concat(df_all)
        
        # Sauvegarder le DataFrame au format CSV et PNG
        filename = result_dir+f'TabN_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}'
        df.to_csv(filename+'.csv')
        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        dfi.export(df, filename+'.png', dpi=300, table_conversion=table_conversion)
        
class CompareMethodsDoFs(CompareMethodsMeshSize):
    def get_total_parameters_of_net(self,u_theta):
        net = u_theta.net
        
        total_params = 0
        for param in net.parameters():
            total_params += param.numel()
        
        return total_params
    
    def get_dofs_at_given_precision_deg_allM(self,given_precision,degree,tab_M=None):
        tab_N = self.get_N_at_given_precision_deg_allM(given_precision,degree,tab_M=tab_M)
        # round the values to the upper integer
        tab_N = {key:round(val) for key,val in tab_N.items()}

        solver = self.ee.solver_type(params=self.ee.params, problem=self.ee.pb_considered, degree=degree, error_degree=None, high_degree=None, save_uref=False)
        
        tab_nb_dofs = []
        for N in tab_N.values():
            solver.set_meshsize(nb_cell=N-1)
            nb_dofs = solver.V.dim()
            tab_nb_dofs.append(nb_dofs)
            
        return tab_N,tab_nb_dofs  
            
    def save_tab_given_precisions_deg_allM(self,u_theta,degree,tab_M=None,tab_given_precision = [1e-3,1e-4],n_params=100):
        result_dir = self.ee.results_dir + "TabDoFs/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        tab_N_prec = {}
        tab_DoFs_prec = {}
        tab_DoFs_prec_nparams = {}
        for given_precision in tab_given_precision:
            print(f"# given_precision={given_precision}")
            tab_N,tab_nb_dofs = self.get_dofs_at_given_precision_deg_allM(given_precision,degree,tab_M)
            for (i,key) in enumerate(tab_N.keys()):
                if not key in tab_DoFs_prec:
                    tab_N_prec[key] = []
                    tab_DoFs_prec[key] = []
                    tab_DoFs_prec_nparams[key] = []
                tab_N_prec[key].append(tab_N[key])
                tab_DoFs_prec[key].append(tab_nb_dofs[i])
                tab_DoFs_prec_nparams[key].append(tab_nb_dofs[i]*n_params)                
            tab_methods = ["PINNs"]+list(tab_N.keys())
            
        total_dofs_PINNs = self.get_total_parameters_of_net(u_theta)
        tab_DoFs_prec["PINNs"] = [total_dofs_PINNs]*len(tab_given_precision)
        tab_N_prec["PINNs"] = [None]*len(tab_given_precision)
                
        df_N = pd.DataFrame(tab_N_prec, index=tab_given_precision, columns=tab_methods)
        df_Dofs = pd.DataFrame(tab_DoFs_prec, index=tab_given_precision, columns=tab_methods)

        df = pd.concat([df_N,df_Dofs],axis=1)
        df.columns = pd.MultiIndex.from_product([["N","DoFs"],tab_methods],names=["type","method"])
        
        # Sauvegarder le DataFrame au format CSV et PNG
        filename = result_dir+f'TabDoFs_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}'
        df.to_csv(filename+'.csv')
        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        dfi.export(df, filename+'.png', dpi=300, table_conversion=table_conversion)
        
        # Compute for n_params

        for method in tab_DoFs_prec_nparams:
            if method != "FEM":
                for i in range(len(tab_DoFs_prec_nparams[method])):
                    tab_DoFs_prec_nparams[method][i] += total_dofs_PINNs
    
        df_dofs_nparams = pd.DataFrame(tab_DoFs_prec_nparams, index=tab_given_precision, columns=tab_methods[1:])
        
        # Sauvegarder le DataFrame au format CSV et PNG
        filename = result_dir+f'TabDoFsParam_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}_nparams{n_params}'
        df_dofs_nparams.to_csv(filename+'.csv')
        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        dfi.export(df_dofs_nparams, filename+'.png', dpi=300, table_conversion=table_conversion)
        
        return df,df_dofs_nparams

    def save_tab_given_precisions_alldeg_allM(self,u_theta,tab_M=None,n_params=100):
        result_dir = self.ee.results_dir + "TabDoFs/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        df_all = {}
        df_dofs_nparams_all = {}
        for degree in self.ee.tab_degree:
            print(f"## degree={degree}")
            tab_given_precision = [10**-(degree+2),10**-(degree+3)]
            df,df_dofs_nparams = self.save_tab_given_precisions_deg_allM(u_theta,degree,tab_M,tab_given_precision=tab_given_precision,n_params=n_params)
            df_all[degree] = df
            df_dofs_nparams_all[degree] = df_dofs_nparams
            
        df = pd.concat(df_all)
        
        # Sauvegarder le DataFrame au format CSV et PNG
        filename = result_dir+f'TabDoFs_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}'
        df.to_csv(filename+'.csv')
        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        dfi.export(df, filename+'.png', dpi=300, table_conversion=table_conversion)
        
        df_dofs_nparams = pd.concat(df_dofs_nparams_all)
        
        # Sauvegarder le DataFrame au format CSV et PNG
        filename = result_dir+f'TabDoFsParam_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_nparams{n_params}'
        df_dofs_nparams.to_csv(filename+'.csv')
        # table_conversion = "chrome"
        table_conversion = "matplotlib"
        dfi.export(df_dofs_nparams, filename+'.png', dpi=300, table_conversion=table_conversion)
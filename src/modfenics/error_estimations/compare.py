import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataframe_image as dfi

from modfenics.error_estimations.error_estimations import ErrorEstimations

class CompareMethods:
    def __init__(self,error_estimations:ErrorEstimations):
        self.ee = error_estimations

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
            
            # Sauvegarder le DataFrame formaté au format CSV
            formatted_df.to_csv(self.ee.results_dir+f'Tab_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.csv')
            
            # Et au format PNG

            # table_conversion = "chrome"
            table_conversion = "matplotlib"
            dfi.export(formatted_df, self.ee.results_dir+f'Tab_case{self.ee.testcase}_v{self.ee.version}_param{self.ee.param_num}_degree{degree}.png', dpi=300, table_conversion=table_conversion)
        
    def save_tab_alldeg_allM(self,tab_M=None):
        for degree in self.ee.tab_degree:
            self.save_tab_deg_allM(degree,tab_M)
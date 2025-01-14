import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from testcases.utils import create_tree,get_random_params,compute_slope
from modfenics.error_estimations.utils import get_solver_type

class GainsEnhancedFEM:
    def __init__(self, n_params, pb_considered, **kwargs):
        # define the problem
        self.n_params = n_params
        self.pb_considered = pb_considered
        self.__infos_from_problem()
        repo_dir = kwargs.get('repo_dir', "./")
        version_str = f"version{self.version}"
        
        self.results_dir = repo_dir + f"/results/fenics/test_{self.dim}D/testcase{self.testcase}/{version_str}/gains/"
        create_tree(self.results_dir)
        
        print(f"## Results directory: {self.results_dir}")
        
        # define the degrees and the number of vertices
        self.high_degree = kwargs.get('high_degree', 10)
        self.error_degree = kwargs.get('error_degree', 4)
        # self.save_fig = kwargs.get('save_fig', False)
        self.tab_nb_vert = kwargs.get('tab_nb_vert', [20,40])
        self.tab_degree = kwargs.get('tab_degree', [1,2,3])
        
        # define if the reference solution is an analytical solution
        self.save_uref = None
        if not self.pb_considered.ana_sol:
            savedir = self.results_dir + "u_ref/"
            create_tree(savedir)
            self.save_uref = [savedir + f"u_ref_{param_num}.npy" for param_num in range(self.n_params)]
            
        # self.plot_result = kwargs.get('plot_result', False)
        
    def __infos_from_problem(self):
        self.dim = self.pb_considered.dim
        self.testcase = self.pb_considered.testcase
        self.version = self.pb_considered.version
        parameter_domain = self.pb_considered.parameter_domain
        self.params = get_random_params(self.n_params,parameter_domain)   
        self.solver_type = get_solver_type(self.dim,self.testcase,self.version)
        
    def read_csv(self,csv_file):
        df_method = pd.read_csv(csv_file)
        tab_h_method = df_method.values[1,1:]
        tab_err_method = df_method.values[2:,1:]
        
        tab_nb_vert_method = df_method.values[0,1:]
        assert all(tab_nb_vert_method) == all(self.tab_nb_vert)
        
        return df_method,tab_h_method,tab_err_method

    def run_errors_deg(self,method,degree,**kwargs):
        new_run = kwargs.get('new_run', False)
        assert method in ["FEM","PINNs","Corr","Mult"], f"method={method} is not implemented"
        
        assert "u_theta" in kwargs if method in ["Corr","Mult"] else True, f"u_theta is required for {method}"
        u_theta = kwargs.get('u_theta')
        
        if method == "Mult":
            assert 'M' in kwargs and 'impose_bc' in kwargs, f"M and impose_bc are required for {method}"
            M = kwargs.get('M')
            impose_bc = kwargs.get('impose_bc')
        else:
            assert not 'M' in kwargs and not 'impose_bc' in kwargs, f"M and impose_bc are not required for {method}"

        # Define the csv_file        
        csv_file = self.results_dir+f"{method}_errors_case{self.testcase}_v{self.version}_degree{degree}"
        if method == "Mult":
            csv_file += f"_M{M}"
            csv_file += '_weak' if not impose_bc else ''
        csv_file += ".csv"
        
        # Read the csv file if it exists (or if not new_run)
        if not new_run and os.path.exists(csv_file):
            print(f"## Read csv file : {csv_file}")
            return self.read_csv(csv_file)

        # Run the error estimation
        print(f"## Run errrors with {method} for degree={degree}")
        # tab_h_method = []
        # tab_err_method = []
        tab_h_method = []
        tab_err_method = np.zeros((self.n_params,len(self.tab_nb_vert)))
        
        solver = self.solver_type(params=self.params, problem=self.pb_considered, degree=degree, error_degree=self.error_degree, high_degree=self.high_degree, save_uref=self.save_uref)
        
        for (j,nb_vert) in enumerate(self.tab_nb_vert):
            print(f"nb_vert={nb_vert}")
            solver.set_meshsize(nb_cell=nb_vert-1)
            tab_h_method.append(np.round(solver.h,3))
            
            for i in range(self.n_params):
                print(i,end=" ")
                if method == "FEM": 
                    _,norme_L2 = solver.fem(i)
                elif method == "PINNs":
                    norme_L2 = solver.pinns(i,u_theta)
                elif method == "Corr":
                    _,_,norme_L2 = solver.corr_add(i,u_theta)
                else:
                    _,_,norme_L2 = solver.corr_mult(i,u_theta,M=M,impose_bc=impose_bc)
                tab_err_method[i,j] = norme_L2
            
        col_names = [(method,str(self.tab_nb_vert[i]),tab_h_method[i]) for i in range(len(self.tab_nb_vert))]
        mi = pd.MultiIndex.from_tuples(col_names, names=["method","n_vert","h"])
        df_method = pd.DataFrame(tab_err_method,columns=mi)
        df_method.to_csv(csv_file)
        
        df_method = pd.DataFrame(tab_err_method,columns=mi)
        df_method.to_csv(csv_file)

        return df_method, tab_h_method, tab_err_method
    
    def run_errors_alldeg(self,method,**kwargs):
        for degree in self.tab_degree:
            self.run_errors_deg(method,degree,**kwargs)    
            
    
    def run_fem_deg(self,degree,new_run=False):
        return self.run_errors_deg("FEM",degree,new_run=new_run)
    
    def run_pinns_deg(self,degree,u_theta,new_run=False):
        return self.run_errors_deg("PINNs",degree,u_theta=u_theta,new_run=new_run)
            
    def run_corr_deg(self,degree,u_theta,new_run=False):
        return self.run_errors_deg("Corr",degree,u_theta=u_theta,new_run=new_run)
        
    def run_mult_deg_M(self,degree,u_theta,M=0.0,impose_bc=True,new_run=False):
        return self.run_errors_deg("Mult",degree,u_theta=u_theta,M=M,impose_bc=impose_bc,new_run=new_run)
        
    
    def run_fem_alldeg(self,new_run=False):
        self.run_errors_alldeg("FEM",new_run=new_run)
        
    def run_pinns_alldeg(self,u_theta,new_run=False):
        self.run_errors_alldeg("PINNs",u_theta=u_theta,new_run=new_run)
        
    def run_corr_alldeg(self,u_theta,new_run=False):
        self.run_errors_alldeg("Corr",u_theta=u_theta,new_run=new_run)
        
    def run_mult_alldeg_M(self,u_theta,M=0.0,impose_bc=True,new_run=False):
        self.run_errors_alldeg("Mult",u_theta=u_theta,M=M,impose_bc=impose_bc,new_run=new_run)
        
    def run_mult_deg_allM(self,degree,u_theta,tab_M,impose_bc=True,new_run=False):
        for M in tab_M:
            self.run_mult_deg_M(degree,u_theta,M=M,impose_bc=impose_bc,new_run=new_run)
            
    def run_mult_alldeg_allM(self,u_theta,tab_M,impose_bc=True,new_run=False):
        for M in tab_M:
            self.run_mult_alldeg_M(u_theta,M=M,impose_bc=impose_bc,new_run=new_run)
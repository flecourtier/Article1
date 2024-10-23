print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import UexExpr
from modfenics.utils import get_utheta_fenics_onV
import dolfin as df

import abc
import time
from pathlib import Path

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 10

current = Path(__file__).parent.parent

#######
# FEM #
#######

class FEMSolver(abc.ABC):
    def __init__(self,params,problem,degree=1,error_degree=4,high_degree=9):
        self.N = None # number of cells
        self.params = params # list of parameters
        self.pb_considered = problem # problem considered
        self.degree = degree # degree of the finite element space
        self.error_degree = error_degree # degree of the error space
        self.high_degree = high_degree # degree of the expression space for f
        
        # To evaluate computational time
        self.times_fem = {}
        self.times_corr_add = {}
        
        # To compute error (overrefined mesh)
        self.N_ex = 500 #5*self.N
        self.mesh_ex,self.V_ex,self.dx_ex = self._create_FEM_domain(self.N_ex+1,self.error_degree) 
        self.h_ex = self.mesh_ex.hmax()
        print("V_ex created with ",self.N_ex+1," vertices and degree ",self.error_degree," : h_ex =",self.h_ex)
        # self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)
            
    @abc.abstractmethod
    def _create_mesh(self,nb_vert):
        pass    

    @abc.abstractmethod
    def _define_fem_system(self,params,u,v):
        pass
    
    @abc.abstractmethod
    def _define_corr_add_system(self):
        pass
    
    def set_meshsize(self,nb_cell):
        self.N = nb_cell # number of cells
        
        self.times_fem[self.N] = {}
        self.times_corr_add[self.N] = {}
        
        # To compute the solution with FEM (standard/correction)
        self.mesh,self.V,self.dx = self._create_FEM_domain(self.N+1,self.degree,save_times=True)
        self.h = self.mesh.hmax()
        print("Mesh created with ",self.N+1," vertices : h =",self.h)
        
        self.V_theta = df.FunctionSpace(self.mesh, "CG", self.high_degree)
        
    def _create_FEM_domain(self,nb_vert,degree,save_times=False):        
        # Construct a cartesian mesh with nb_vert-1 cells in each direction
        start = time.time()
        mesh = self._create_mesh(nb_vert)
        end = time.time()

        if save_times:
            if print_time:
                print("Time to generate mesh: ", end-start)
            self.times_fem[self.N]["mesh"] = end-start
            self.times_corr_add[self.N]["mesh"] = end-start
        
        V = df.FunctionSpace(mesh, "CG", degree)
        dx = df.Measure("dx", domain=mesh)

        return mesh, V, dx
    
    def fem(self, i):
        assert self.N is not None
        params = self.params[i]
        
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        # Declaration of the variationnal problem
        start = time.time()        
        A,L = self._define_fem_system(params,u,v)
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem[self.N]["assemble"] = end-start

        # Resolution of the linear system
        start = time.time()
        sol = df.Function(self.V)
        df.solve(A,sol.vector(),L)
        end = time.time()
        
        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem[self.N]["solve"] = end-start

        # Compute the error
        start = time.time()
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        uex_Vex = df.interpolate(u_ex,self.V_ex)
        sol_Vex = df.interpolate(sol,self.V_ex)
        norme_L2 = (df.assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (df.assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        end = time.time()
        
        if print_time:
            print("Time to compute the error :",end-start)
        self.times_fem[self.N]["error"] = end-start
        
        return sol,norme_L2

    def corr_add(self, i, u_PINNs): #phi_tild, lap_phi_tild):
        assert self.N is not None
        params = self.params[i]
        
        from time import sleep
        
        u_theta_V = get_utheta_fenics_onV(self.V,self.params[i],u_PINNs)      
        
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        
        # Declaration of the variationnal problem
        start = time.time()
        A,L = self._define_corr_add_system(params,u,v,u_PINNs)
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_corr_add[self.N]["assemble"] = end-start

        # Resolution of the linear system
        start = time.time()
        C_tild = df.Function(self.V)
        df.solve(A,C_tild.vector(),L)
        
        sol = df.Function(self.V)
        sol.vector()[:] = C_tild.vector()[:] + u_theta_V.vector()[:]
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add[self.N]["solve"] = end-start

        # Compute the error
        u_theta_Vex = get_utheta_fenics_onV(self.V_ex,self.params[i],u_PINNs)
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        uex_Vex = df.interpolate(u_ex,self.V_ex) 
        C_Vex = df.interpolate(C_tild,self.V_ex)
        sol_Vex = df.Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+u_theta_Vex.vector()[:]
        
        norme_L2 = (df.assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (df.assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
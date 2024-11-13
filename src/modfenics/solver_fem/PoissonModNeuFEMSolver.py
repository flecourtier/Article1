print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import get_f_expr,get_uex_expr
from modfenics.solver_fem.FEMSolver import FEMSolver
from modfenics.utils import get_laputheta_fenics_fromV,get_utheta_fenics_onV,get_gradutheta_fenics_fromV
import dolfin as df

import numpy as np
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

from modfenics.solver_fem.GeometryFEMSolver import DonutFEMSolver

class PoissonModNeuFEMSolver(FEMSolver):    
    def _define_fem_system(self,params,u,v,V_solve):
        u_ex = get_uex_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        
        # Impose Neumann boundary conditions
        normals = df.FacetNormal(V_solve.mesh())
        h = df.inner(df.grad(u_ex),normals)
                
        dx = df.Measure("dx", domain=V_solve.mesh())
        ds = df.Measure("ds", domain=V_solve.mesh())
        
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner(df.grad(u),df.grad(v)) * dx + u*v*dx
        l = f_expr * v * dx + h * v * ds

        A = df.assemble(a)
        L = df.assemble(l)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        u_ex = get_uex_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        
        f_expr_Vtheta = df.interpolate(f_expr,self.V_theta)
        u_theta_Vtheta = get_utheta_fenics_onV(self.V_theta,params,u_PINNs)
        lap_utheta_Vtheta = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = f_expr_Vtheta.vector()[:] + lap_utheta_Vtheta.vector()[:] - u_theta_Vtheta.vector()[:]
        
        # Impose Neumann boundary conditions
        gradu_theta_Vtheta = df.grad(u_theta_Vtheta)
        # gradu_theta_Vtheta = get_gradutheta_fenics_fromV(self.V_theta,params,u_PINNs)
        normals = df.FacetNormal(V_solve.mesh())
        h = df.inner(df.grad(u_ex),normals)
        h_tild = h - df.inner(gradu_theta_Vtheta,normals)
        
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        ds = df.Measure("ds", domain=V_solve.mesh())
        
        
        a = df.inner(df.grad(u),df.grad(v)) * dx + u*v*dx
        l = f_tild * v * dx + h_tild * v * ds


        A = df.assemble(a)
        L = df.assemble(l)
        
        return A,L
    
    def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M):
        pass

class PoissonModNeuDonutFEMSolver(PoissonModNeuFEMSolver,DonutFEMSolver):
    pass
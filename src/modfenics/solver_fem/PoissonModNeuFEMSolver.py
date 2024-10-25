print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import FExpr,UexExpr
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
        u_ex = UexExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        
        # Impose Neumann boundary conditions
        ########## A REVOIR
        normals = df.FacetNormal(V_solve.mesh())
        normals_V = df.interpolate(normals,V_solve)
        u_ex_V = df.interpolate(u_ex,V_solve)
        h = df.inner(df.grad(u_ex_V),normals_V)
                
        dx = df.Measure("dx", domain=V_solve.mesh())
        ds = df.Measure("ds", domain=V_solve.mesh())
        
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner(df.grad(u),df.grad(v)) * dx + u*v*dx
        l = f_expr * v * dx + h * v * ds

        A = df.assemble(a)
        L = df.assemble(l)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        u_ex = UexExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        lap_utheta = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        fexpr_inter = df.interpolate(f_expr,self.V_theta)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = fexpr_inter.vector()[:] + lap_utheta.vector()[:] # div(grad(phi_tild))

        # Impose Neumann boundary conditions
        normals = df.FacetNormal(self.mesh)
        normals_V = df.interpolate(normals,V_solve)
        u_ex_V = df.interpolate(u_ex,V_solve)
        h = df.inner(df.grad(u_ex_V),normals_V)
        grad_u_theta_V = get_gradutheta_fenics_fromV(self.V_theta,params,u_PINNs)
        h_tild = h - df.inner(grad_u_theta_V,normals_V)
        
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
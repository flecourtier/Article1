print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import get_f_expr,get_uex_expr
from modfenics.solver_fem.NonLinearFEMSolver import NonLinearFEMSolver
from modfenics.utils import get_laputheta_fenics_fromV,get_utheta_fenics_onV,get_gradutheta_fenics_fromV
from testcases.geometry.geometry_2D import Circle
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

from modfenics.solver_fem.GeometryFEMSolver import CircleFEMSolver

class NonLinearPoissonDirFEMSolver(NonLinearFEMSolver):    
    def _define_fem_system(self,params,u,v,V_solve):
        boundary = "on_boundary"
        assert isinstance(self.pb_considered.geometry, Circle)
        
        g = df.Constant("0.0")
        # g = df.Constant("1.0")
        
        bc = df.DirichletBC(V_solve, g, boundary)
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner((1+u**2)*df.grad(u), df.grad(v)) * dx
        l = f_expr * v * dx

        F = a - l

        return F,bc

    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        boundary = "on_boundary"
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        f_expr_Vtheta = df.interpolate(f_expr,self.V_theta)
        u_theta_Vtheta = get_utheta_fenics_onV(self.V_theta,params,u_PINNs)

        dx = df.Measure("dx", domain=V_solve.mesh())

        g = df.Constant("0.0")
        bc = df.DirichletBC(V_solve, g, boundary)
        a = df.inner((1+(u_theta_Vtheta+u)**2)*df.grad(u_theta_Vtheta+u), df.grad(v)) * dx
        l = f_expr_Vtheta * v * dx

        F = a - l

        return F,bc
    
    def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M,impose_bc):
        pass
    
class NonLinearPoissonDirCircleFEMSolver(NonLinearPoissonDirFEMSolver,CircleFEMSolver):
    pass


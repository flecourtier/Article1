print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import FExpr,UexExpr
from modfenics.solver_fem.FEMSolver import FEMSolver
from modfenics.utils import get_laputheta_fenics_fromV,get_utheta_fenics_onV
from testcases.geometry.geometry_2D import Square,Donut
from testcases.geometry.geometry_1D import Line
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

from modfenics.solver_fem.GeometryFEMSolver import LineFEMSolver,SquareFEMSolver,DonutFEMSolver

class PoissonDirFEMSolver(FEMSolver):    
    def _define_fem_system(self,params,u,v,V_solve):
        boundary = "on_boundary"
        print(self.pb_considered.geometry)
        if isinstance(self.pb_considered.geometry, (Square,Line)):
            g = df.Constant("0.0")
        elif isinstance(self.pb_considered.geometry, Donut):
            u_ex = UexExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
            g = df.interpolate(u_ex,V_solve)
        else:
            raise ValueError("Geometry not recognized")
        bc = df.DirichletBC(V_solve, g, boundary)
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner(df.grad(u), df.grad(v)) * dx
        l = f_expr * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        lap_utheta = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        
        boundary = "on_boundary"
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        fexpr_inter = df.interpolate(f_expr,self.V_theta)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = fexpr_inter.vector()[:] + lap_utheta.vector()[:] # div(grad(phi_tild))

        dx = df.Measure("dx", domain=V_solve.mesh())

        if isinstance(self.pb_considered.geometry, (Square,Line)):
            g = df.Constant("0.0")
        elif isinstance(self.pb_considered.geometry, Donut):
            u_ex = UexExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
            u_ex_V = df.interpolate(u_ex,V_solve)
            u_theta_V = get_utheta_fenics_onV(V_solve,params,u_PINNs)
            g = u_ex_V - u_theta_V
        else:
            raise ValueError("Geometry not recognized")
        bc = df.DirichletBC(V_solve, g, boundary)
        a = df.inner(df.grad(u), df.grad(v)) * dx
        l = f_tild * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M):
        assert isinstance(self.pb_considered.geometry, Line)
        
        u_theta_V = get_utheta_fenics_onV(self.V_theta,params,u_PINNs) 
        u_theta_M_V = df.Function(self.V_theta)
        u_theta_M_V.vector()[:] = u_theta_V.vector()[:] + M
        
        boundary = "on_boundary"
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        dx = df.Measure("dx", domain=V_solve.mesh())

        g = df.Constant(1.0)
        bc = df.DirichletBC(V_solve, g, boundary)
        
        a = df.inner(df.grad(u_theta_M_V * u), df.grad(u_theta_M_V * v)) * dx
        l = f_expr * u_theta_M_V * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
class PoissonDirLineFEMSolver(PoissonDirFEMSolver,LineFEMSolver):
    pass

class PoissonDirSquareFEMSolver(PoissonDirFEMSolver,SquareFEMSolver):
    pass

class PoissonDirDonutFEMSolver(PoissonDirFEMSolver,DonutFEMSolver):
    pass


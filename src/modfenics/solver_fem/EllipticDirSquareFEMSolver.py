print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import FExpr,AnisotropyExpr
from modfenics.solver_fem.FEMSolver import FEMSolver
from modfenics.utils import get_divmatgradutheta_fenics_fromV
from testcases.geometry.geometry_2D import Square
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

from modfenics.solver_fem.PoissonDirSquareFEMSolver import SquareFEMSolver

class EllipticDirFEMSolver(FEMSolver):    
    def _define_fem_system(self,params,u,v,V_solve):
        boundary = "on_boundary"
        g = df.Constant("0.0")
        bc = df.DirichletBC(V_solve, g, boundary)
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered) 
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner(mat*df.grad(u), df.grad(v)) * dx
        l = f_expr * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        mat_inter = df.interpolate(mat,self.V_theta)
        print(mat_inter.vector()[:].shape)
        divmatgradutheta = get_divmatgradutheta_fenics_fromV(self.V_theta,params,u_PINNs,mat_inter)
        
        boundary = "on_boundary"
        f_expr = FExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        fexpr_inter = df.interpolate(f_expr,self.V_theta)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = fexpr_inter.vector()[:] + divmatgradutheta.vector()[:] # div(mat*grad(phi_tild))

        dx = df.Measure("dx", domain=V_solve.mesh())

        g = df.Constant(0.0)
        bc = df.DirichletBC(V_solve, g, boundary)
        mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered) 
        a = df.inner(mat*df.grad(u), df.grad(v)) * dx
        l = f_tild * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
class EllipticDirSquareFEMSolver(EllipticDirFEMSolver,SquareFEMSolver):
    pass
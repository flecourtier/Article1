print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import FExpr
from modfenics.solver_fem.FEMSolver import FEMSolver
from modfenics.utils import get_laputheta_fenics_fromV
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

class SquareFEMSolver(FEMSolver):    
    def _create_mesh(self,nb_vert):
        # check if pb_considered is instance of Square class
        assert isinstance(self.pb_considered.geometry, Square)
        
        box = np.array(self.pb_considered.geometry.box)
        mesh = df.RectangleMesh(df.Point(box[0,0], box[1,0]), df.Point(box[0,1], box[1,1]), nb_vert - 1, nb_vert - 1)
        
        return mesh

class PoissonDirFEMSolver(FEMSolver):    
    def _define_fem_system(self,params,u,v):
        boundary = "on_boundary"
        g = df.Constant("0.0")
        bc = df.DirichletBC(self.V, g, boundary)
        
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        a = df.inner(df.grad(u), df.grad(v)) * self.dx
        l = f_expr * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs):
        lap_utheta = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        
        boundary = "on_boundary"
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        fexpr_inter = df.interpolate(f_expr,self.V_theta)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = fexpr_inter.vector()[:] + lap_utheta.vector()[:] # div(grad(phi_tild))

        g = df.Constant(0.0)
        bc = df.DirichletBC(self.V, g, boundary)
        a = df.inner(df.grad(u), df.grad(v)) * self.dx
        l = f_tild * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
class PoissonDirSquareFEMSolver(PoissonDirFEMSolver,SquareFEMSolver):
    pass
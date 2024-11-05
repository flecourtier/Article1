print_time = True

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import get_f_expr,AnisotropyExpr
from modfenics.solver_fem.FEMSolver import FEMSolver
from modfenics.utils import get_divmatgradutheta_fenics_fromV, get_utheta_fenics_onV,get_gradutheta_fenics_fromV,get_laputheta_fenics_fromV
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

from modfenics.solver_fem.GeometryFEMSolver import SquareFEMSolver,LineFEMSolver
from testcases.geometry.geometry_2D import Square

class EllipticDirFEMSolver(FEMSolver):    
    def _define_fem_system(self,params,u,v,V_solve):
        boundary = "on_boundary"
        
        g = df.Constant("0.0")
        bc = df.DirichletBC(V_solve, g, boundary)
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered) 
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner(mat*df.grad(u), df.grad(v)) * dx
        l = f_expr * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        divmatgradutheta = get_divmatgradutheta_fenics_fromV(self.V_theta,params,u_PINNs,self.pb_considered.anisotropy_matrix)
        
        boundary = "on_boundary"
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        get_f_expr_inter = df.interpolate(f_expr,self.V_theta)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = get_f_expr_inter.vector()[:] + divmatgradutheta.vector()[:] # div(mat*grad(phi_tild))

        dx = df.Measure("dx", domain=V_solve.mesh())

        g = df.Constant("0.0")
        bc = df.DirichletBC(V_solve, g, boundary)
        
        mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered) 
        a = df.inner(mat*df.grad(u), df.grad(v)) * dx
        l = f_tild * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M,impose_bc):
        assert isinstance(self.pb_considered.geometry, Square)
        assert impose_bc
        
        u_theta_Vtheta = get_utheta_fenics_onV(self.V_theta,params,u_PINNs) 
        u_theta_M_Vtheta = df.Function(self.V_theta)
        u_theta_M_Vtheta.vector()[:] = u_theta_Vtheta.vector()[:] + M
        
        boundary = "on_boundary"
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        dx = df.Measure("dx", domain=V_solve.mesh())

        g = df.Constant(1.0)
        bc = df.DirichletBC(V_solve, g, boundary)
        
        mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered) 
        a = df.inner(mat*df.grad(u_theta_M_Vtheta * u), df.grad(u_theta_M_Vtheta * v)) * dx
        l = f_expr * u_theta_M_Vtheta * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    # def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M):
    #     assert isinstance(self.pb_considered.geometry, Square)
    #     u_theta_V = get_utheta_fenics_onV(V_solve,params,u_PINNs) 
    #     u_theta_M_V = df.Function(V_solve)
    #     u_theta_M_V.vector()[:] = u_theta_V.vector()[:] + M
        
    #     grad_u_theta_V = get_gradutheta_fenics_fromV(V_solve,params,u_PINNs) # same as grad_u_theta_M_V
        
    #     boundary = "on_boundary"
    #     f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
    #     dx = df.Measure("dx", domain=V_solve.mesh())

    #     g = df.Constant(1.0)
    #     bc = df.DirichletBC(V_solve, g, boundary)
        
    #     mat = AnisotropyExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered) 
    #     a = df.inner(mat * (grad_u_theta_V * u + u_theta_M_V * df.grad(u)), df.grad(grad_u_theta_V * v + u_theta_M_V * df.grad(v))) * dx
    #     l = f_expr * u_theta_M_V * v * dx

    #     A = df.assemble(a)
    #     L = df.assemble(l)
    #     bc.apply(A, L)
        
    #     return A,L
    
class EllipticDirSquareFEMSolver(EllipticDirFEMSolver,SquareFEMSolver):
    pass

class Elliptic1DDirFEMSolver(FEMSolver):
    def _define_fem_system(self,params,u,v,V_solve):
        boundary = "on_boundary"
        r,Pe = params
        
        g = df.Constant("0.0")
        bc = df.DirichletBC(V_solve, g, boundary)
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        f_expr = df.Constant(r)
        a = df.grad(u)[0] * v * dx + 1.0/Pe * df.inner(df.grad(u), df.grad(v)) * dx
        l = f_expr * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        boundary = "on_boundary"
        r,Pe = params
        
        u_theta_x = get_gradutheta_fenics_fromV(self.V_theta,params,u_PINNs)
        u_theta_xx = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        
        f_expr = df.Constant(r)
        get_f_expr_inter = df.interpolate(f_expr,self.V_theta)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = get_f_expr_inter.vector()[:] - u_theta_x.vector()[:] + 1.0/Pe * u_theta_xx.vector()[:] 

        dx = df.Measure("dx", domain=V_solve.mesh())

        g = df.Constant("0.0")
        bc = df.DirichletBC(V_solve, g, boundary)
        
        a = df.grad(u)[0] * v * dx + 1.0/Pe * df.inner(df.grad(u), df.grad(v)) * dx
        l = f_tild * v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)
        
        return A,L
    
    def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M,impose_bc):
        boundary = "on_boundary"
        r,Pe = params
        
        # u_theta_x = get_gradutheta_fenics_fromV(self.V_theta,params,u_PINNs)
        # u_theta_xx = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        
        u_theta_Vtheta = get_utheta_fenics_onV(self.V_theta,params,u_PINNs) 
        u_theta_M_Vtheta = df.Function(self.V_theta)
        u_theta_M_Vtheta.vector()[:] = u_theta_Vtheta.vector()[:] + M
        
        dx = df.Measure("dx", domain=V_solve.mesh())

        if impose_bc:
            g = df.Constant(1.0)
            bc = df.DirichletBC(V_solve, g, boundary)
        
        f_expr = df.Constant(r)
        a = df.grad(u_theta_M_Vtheta*u)[0] * u_theta_M_Vtheta*v * dx + 1.0/Pe * df.inner(df.grad(u_theta_M_Vtheta*u), df.grad(u_theta_M_Vtheta*v)) * dx
        l = f_expr * u_theta_M_Vtheta*v * dx

        A = df.assemble(a)
        L = df.assemble(l)
        if impose_bc:
            bc.apply(A, L)
        
        return A,L

class Elliptic1DDirLineFEMSolver(Elliptic1DDirFEMSolver,LineFEMSolver):
    pass
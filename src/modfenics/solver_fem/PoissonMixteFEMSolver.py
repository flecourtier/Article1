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

class PoissonMixteFEMSolver(FEMSolver):    
    def _define_fem_system(self,params,u,v,V_solve):
        u_ex = get_uex_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        
        # Impose Dirichlet boundary conditions
        # g_E = GExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        u_ex_V = df.interpolate(u_ex,V_solve)
        g_E = u_ex_V
        R_mid = (self.pb_considered.geometry.bigcircle.radius+self.pb_considered.geometry.hole.radius)/2.0
        def boundary_D(x,on_boundary):
            return on_boundary and x[0]**2+x[1]**2>R_mid**2 
        bc_ext = df.DirichletBC(V_solve, g_E, boundary_D)       
        
        # Impose Robin boundary conditions
        # h_I = GRExpr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        normals = df.FacetNormal(V_solve.mesh())
        h_I = df.inner(df.grad(u_ex),normals) + u_ex
        class BoundaryN(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0]**2+x[1]**2<R_mid**2
        boundary_N = df.MeshFunction("size_t", V_solve.mesh(), V_solve.mesh().topology().dim()-1)
        bcN = BoundaryN()
        bcN.mark(boundary_N, 0)
        ds_int = df.Measure('ds', domain=V_solve.mesh(), subdomain_data=boundary_N)
                
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        f_expr = get_f_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        a = df.inner(df.grad(u),df.grad(v)) * dx + u*v*ds_int
        l = f_expr * v * dx + h_I * v * ds_int

        A = df.assemble(a)
        L = df.assemble(l)
        bc_ext.apply(A, L)
        
        return A,L
    
    def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
        f_expr = get_f_expr(params, degree=self.high_degree, domain=self.V_theta.mesh(), pb_considered=self.pb_considered)
        u_ex = get_uex_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)


        # f_tild = f_expr + df.div(df.grad(u_theta_Vtheta))
        f_expr_Vtheta = df.interpolate(f_expr,self.V_theta)
        lap_utheta_Vtheta = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
        f_tild = df.Function(self.V_theta)
        f_tild.vector()[:] = f_expr_Vtheta.vector()[:] + lap_utheta_Vtheta.vector()[:] # div(grad(phi_tild))
            
        # Impose Dirichlet boundary conditions (g_tild = 0 sur Gamma_D)
        u_ex_V = df.interpolate(u_ex, self.V) 
        u_theta_V = get_utheta_fenics_onV(V_solve,params,u_PINNs)
        g_E = u_ex_V
        g_tild = g_E - u_theta_V
        R_mid = (self.pb_considered.geometry.bigcircle.radius+self.pb_considered.geometry.hole.radius)/2.0
        def boundary_D(x,on_boundary):
            return on_boundary and x[0]**2+x[1]**2>R_mid**2 
        bc_ext = df.DirichletBC(self.V, g_tild, boundary_D)
        
        # Impose Robin boundary conditions
        u_theta_Vtheta = get_utheta_fenics_onV(self.V_theta,params,u_PINNs)
        gradu_theta_Vtheta = df.grad(u_theta_Vtheta)
        # gradu_theta_Vtheta = get_gradutheta_fenics_fromV(V_solve,params,u_PINNs)
        normals = df.FacetNormal(V_solve.mesh())
        h_I = df.inner(df.grad(u_ex),normals) + u_ex
        h_tild = h_I - (df.inner(gradu_theta_Vtheta,normals) + u_theta_Vtheta)
        class BoundaryN(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0]**2+x[1]**2<R_mid**2
        boundary_N = df.MeshFunction("size_t", V_solve.mesh(), V_solve.mesh().topology().dim()-1)
        bcN = BoundaryN()
        bcN.mark(boundary_N, 1)
        ds_int = df.Measure('ds', domain=V_solve.mesh(), subdomain_data=boundary_N)
        
        dx = df.Measure("dx", domain=V_solve.mesh())
        
        a = df.inner(df.grad(u),df.grad(v)) * dx + u*v*ds_int(1)
        l = f_tild * v * dx + h_tild * v * ds_int(1)

        A = df.assemble(a)
        L = df.assemble(l)
        bc_ext.apply(A, L)
        
        return A,L
    
    # def _define_corr_add_system(self,params,u,v,u_PINNs,V_solve):
    #     u_ex = get_uex_expr(params, degree=self.high_degree, domain=V_solve.mesh(), pb_considered=self.pb_considered)
        
    #     f_expr = get_f_expr(params, degree=self.high_degree, domain=self.V_theta.mesh(), pb_considered=self.pb_considered)
    #     f_expr_Vtheta = df.interpolate(f_expr,self.V_theta)
    #     lap_utheta_Vtheta = get_laputheta_fenics_fromV(self.V_theta,params,u_PINNs)
    #     f_tild = df.Function(self.V_theta)
    #     f_tild.vector()[:] = f_expr_Vtheta.vector()[:] + lap_utheta_Vtheta.vector()[:] # div(grad(phi_tild))

    #     # Impose Dirichlet boundary conditions (g_tild = 0 sur Gamma_D)
    #     u_ex_V = df.interpolate(u_ex, V_solve) 
    #     u_theta_V = get_utheta_fenics_onV(V_solve,params,u_PINNs)
    #     g_tild = u_ex_V - u_theta_V
    #     R_mid = (self.pb_considered.geometry.bigcircle.radius+self.pb_considered.geometry.hole.radius)/2.0
    #     def boundary_D(x,on_boundary):
    #         return on_boundary and x[0]**2+x[1]**2>R_mid**2 
    #     bc_ext = df.DirichletBC(V_solve, g_tild, boundary_D)
        
    #     # Impose Robin boundary conditions
    #     ########## A REVOIR
    #     normals = df.FacetNormal(V_solve.mesh())
    #     # normals_V = df.interpolate(normals,V_solve)
    #     # grad_utheta_V = get_gradutheta_fenics_fromV(V_solve,params,u_PINNs)
    #     grad_utheta_V = df.grad(u_theta_V)
    #     h_I = df.inner(df.grad(u_ex_V),normals) + u_ex_V
    #     h_tild = h_I - (df.inner(grad_utheta_V,normals) + u_theta_V)
    #     class BoundaryN(df.SubDomain):
    #         def inside(self, x, on_boundary):
    #             return on_boundary and x[0]**2+x[1]**2<R_mid**2
    #     boundary_N = df.MeshFunction("size_t", V_solve.mesh(), V_solve.mesh().topology().dim()-1)
    #     bcN = BoundaryN()
    #     bcN.mark(boundary_N, 1)
    #     ds_int = df.Measure('ds', domain=V_solve.mesh(), subdomain_data=boundary_N)
        
    #     dx = df.Measure("dx", domain=V_solve.mesh())
        
    #     a = df.inner(df.grad(u),df.grad(v)) * dx + u*v*ds_int(1)
    #     l = f_tild * v * dx + h_tild * v * ds_int(1)

    #     A = df.assemble(a)
    #     L = df.assemble(l)
    #     bc_ext.apply(A, L)
        
    #     return A,L
    
    def _define_corr_mult_system(self,params,u,v,u_PINNs,V_solve,M):
        pass

class PoissonMixteDonutFEMSolver(DonutFEMSolver,PoissonMixteFEMSolver):
    pass
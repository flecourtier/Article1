###########
# Imports #
###########

from modules.fenics_expressions_1D import *
from modules.geometry_1D import Line

from dolfin import *
import dolfin as df
import mshr
import time
import numpy as np
from pathlib import Path

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 10

current = Path(__file__).parent.parent

#######
# FEM #
#######

class FEMSolver():
    def __init__(self,nb_cell,params,problem,degree=1,high_degree=10):
        self.N = nb_cell # check ?
        self.params = params
        self.pb_considered = problem
        self.degree = degree
        self.mesh,self.V,self.dx = self.__create_FEM_domain()
        
        # to compute error
        self.high_degree = high_degree 
        self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)

    def __create_FEM_domain(self):
        assert isinstance(self.pb_considered.geometry, Line)
        
        nb_vert = self.N+1
        
        box = np.array(self.pb_considered.geometry.box)
        mesh = IntervalMesh(nb_vert - 1, box[0,0], box[0,1])
        
        V = FunctionSpace(mesh, "CG", self.degree)
        dx = Measure("dx", domain=mesh)
        
        self.h = mesh.hmax()
        print("hmax = ",self.h)
        
        return mesh, V, dx
    
    def fem(self, i):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        
        g = Constant("0.0")
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(u), grad(v)) * self.dx
        l = f_expr * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        sol = Function(self.V)
        solve(A,sol.vector(),L)

        uex_Vex = interpolate(u_ex,self.V_ex)
        sol_Vex = interpolate(sol,self.V_ex)
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def corr_add(self, i, phi_tild):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        f_tild = f_expr + div(grad(phi_tild))

        g = Constant(0.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(u), grad(v)) * self.dx
        l = f_tild * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        C_tild = Function(self.V)
        solve(A,C_tild.vector(),L)
        sol = C_tild + phi_tild

        uex_Vex = interpolate(u_ex,self.V_ex)        
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+phi_tild.vector()[:]
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
    
    def corr_mult(self, i, phi_tild, M=0.0):
        boundary = "on_boundary"

        params = self.params[i]
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)

        phi_tild_M = Function(self.V_ex)
        phi_tild_M.vector()[:] = phi_tild.vector()[:] + M
        
        g = Constant(1.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = inner(grad(phi_tild_M*u), grad(v)) * self.dx
        l = f_expr * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        C_tild = Function(self.V)
        solve(A,C_tild.vector(),L)
        sol = phi_tild_M * C_tild - M
        
        uex_Vex = interpolate(u_ex,self.V_ex)
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = C_Vex.vector()[:] * phi_tild_M.vector()[:]
        sol_Vex = sol_Vex - M
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
    
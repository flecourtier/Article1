# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

from fenics_expressions import *
from geometry import Square

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

current = Path(__file__).parent.parent#.parent.parent

#######
# FEM #
#######

class FEMSolver():
    def __init__(self,nb_cell,params,problem,degree=1):
        self.N = nb_cell
        self.params = params
        self.pb_considered = problem
        self.degree = degree
        self.high_degree = 10 # to compute error
        
        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()
        
        self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)

    def __create_FEM_domain(self):
        nb_vert = self.N+1

        # check if pb_considered is instance of Square class
        if isinstance(self.pb_considered.geometry, Square):
            box = np.array(self.pb_considered.geometry.box)
            start = time.time()
            mesh = RectangleMesh(Point(box[0,0], box[1,0]), Point(box[0,1], box[1,1]), nb_vert - 1, nb_vert - 1)
            end = time.time()

            if print_time:
                print("Time to generate mesh: ", end-start)
            self.times_fem["mesh"] = end-start
            self.times_corr_add["mesh"] = end-start
        else:
            raise ValueError("Geometry not implemented")
        
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
            
        if cd=="homo":
            g = Constant("0.0")
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        
        start = time.time()

        a = inner(grad(u), grad(v)) * self.dx
        l = f_expr * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem["assemble"] = end-start

        sol = Function(self.V)

        start = time.time()
        solve(A,sol.vector(),L)
        # solve(a==l, sol, bcs=bc)
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem["solve"] = end-start

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
        # g = Function(self.V)
        # phi_tild_inter = interpolate(phi_tild, self.V)
        # g.vector()[:] = (phi_tild_inter.vector()[:])        
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        start = time.time()
        a = inner(grad(u), grad(v)) * self.dx
        l = f_tild * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_corr_add["assemble"] = end-start

        C_tild = Function(self.V)

        start = time.time()
        solve(A,C_tild.vector(),L)
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add["solve"] = end-start

        sol = C_tild + phi_tild

        uex_Vex = interpolate(u_ex,self.V_ex)
        
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+phi_tild.vector()[:]
        
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
    
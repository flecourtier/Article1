# homogeneous = True
cd = "homo"
print_time=True

###########
# Imports #
###########

from modules.fenics_expressions import *
from modules.geometry_3D import Cube

from dolfin import *
import dolfin as df
# import mshr
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

prm = parameters["krylov_solver"]
prm["absolute_tolerance"] = 1e-13
prm["relative_tolerance"] = 1e-13

#######
# FEM #
#######

class FEMSolver():
    def __init__(self,nb_cell,params,problem,degree=1,high_degree=10):
        self.N = nb_cell
        self.params = params
        self.pb_considered = problem
        self.degree = degree
        self.high_degree = high_degree # to compute error
        
        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()
        
        self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)

    def __create_FEM_domain(self):
        nb_vert = self.N+1

        # check if pb_considered is instance of Square class
        if isinstance(self.pb_considered.geometry, Cube):
            box = np.array(self.pb_considered.geometry.box)
            start = time.time()
            mesh = BoxMesh(Point(box[0,0], box[1,0], box[2,0]), Point(box[0,1], box[1,1], box[2,1]), nb_vert - 1, nb_vert - 1, nb_vert - 1)
            # mesh = RectangleMesh(Point(box[0,0], box[1,0]), Point(box[0,1], box[1,1]), nb_vert - 1, nb_vert - 1)
            end = time.time()
            
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plot(mesh)
            # plt.show()

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
    
    def fem(self, i, iter_solver=False):
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
        # print(A.size(0),A.size(1))
        # print("nnz : ",A.nnz())
        bc.apply(A, L)

        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem["assemble"] = end-start

        sol = Function(self.V)

        start = time.time()
        if not iter_solver:
            solve(A,sol.vector(),L)
        else:
            # df.solve(a==l, sol, bcs=bc, solver_parameters={"linear_solver": "cg","preconditioner":"hypre_amg"})
            solve(A,sol.vector(),L,"cg","hypre_amg")
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
        # start = time.time()
        # a = inner(grad(u), grad(v)) * self.dx
        # l = f_tild * v * self.dx

        # A = df.assemble(a)
        # L = df.assemble(l)
        # bc.apply(A, L)

        start = time.time()
        
        start_a = time.time()
        a = inner(grad(u), grad(v)) * self.dx
        A = df.assemble(a)
        end_a = time.time()
        
        start_b = time.time()
        l = f_tild * v * self.dx
        L = df.assemble(l)
        end_b = time.time()
        
        start_bc = time.time()        
        bc.apply(A, L)
        end_bc = time.time()

        end = time.time()

        if print_time:
            print("Time to assemble the matrix A : ",end_a-start_a)
            print("Time to assemble the vector b : ",end_b-start_b)
            print("Time to impose Dirichlet BC : ",end_bc-start_bc)
            print("Time to construct the sytem : ",end-start)
        self.times_corr_add["system"] = end-start
        self.times_corr_add["assemble_A"] = end_a-start_a
        self.times_corr_add["assemble_b"] = end_b-start_b
        self.times_corr_add["impose_BC"] = end_bc-start_bc

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
    
## SOLVEUR - Cas test 5 - Donut

# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import *
from testcases.geometry.geometry_2D import Donut

from dolfin import *
import dolfin as df
import mshr
import time
import numpy as np
from pathlib import Path
# from petsc4py import PETSc

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
    def __init__(self,nb_cell,params,problem,degree=1,high_degree=10):
        self.N = nb_cell
        self.params = params
        self.pb_considered = problem
        self.degree = degree
        self.high_degree = high_degree # to compute error
        
        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()
        self.ds = Measure("ds", domain=self.mesh)
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plot(self.mesh)
        # plt.show()
        
        self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)

    def __create_FEM_domain(self):
        nb_vert = self.N+1

        # check if pb_considered is instance of Square class        
        if isinstance(self.pb_considered.geometry, Donut):
            bigcenter = self.pb_considered.geometry.bigcircle.center
            bigradius = self.pb_considered.geometry.bigcircle.radius
            smallcenter = self.pb_considered.geometry.hole.center
            smallradius = self.pb_considered.geometry.hole.radius
            box = np.array(self.pb_considered.geometry.box)
            
            start = time.time()
            bigcircle = mshr.Circle(df.Point(bigcenter[0],bigcenter[1]), bigradius)
            hole = mshr.Circle(df.Point(smallcenter[0],smallcenter[1]), smallradius)
            domain = bigcircle-hole
            # domain.set_subdomain(1, mshr.Rectangle(df.Point(-1.0, 0.0), df.Point(1.0, 1.0)))
            end = time.time()
            
            mesh_macro = RectangleMesh(Point(box[0,0], box[1,0]), Point(box[0,1], box[1,1]), nb_vert, nb_vert)
            h_macro = mesh_macro.hmax()
            H = int(nb_vert/3)
            mesh = mshr.generate_mesh(domain,H)
            h = mesh.hmax()
            while h > h_macro:
                H += 1
                start2 = time.time()
                mesh = mshr.generate_mesh(domain,H)
                end2 = time.time()
                h = mesh.hmax()

            if print_time:
                print("Time to generate mesh: ", end-start + end2-start2)
            self.times_fem["mesh"] = end-start + end2-start2
            self.times_corr_add["mesh"] = end-start + end2-start2
        else:
            raise ValueError("Geometry not implemented")
        
        V = FunctionSpace(mesh, "CG", self.degree)
        dx = Measure("dx", domain=mesh)
        
        self.h = mesh.hmax()
        print("hmax = ",self.h)

        return mesh, V, dx
        
    def fem(self, i):
        # boundary = "on_boundary"
        params = self.params[i]
        
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)  
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        
        # Impose Neumann boundary conditions
        h = df.inner(grad(u_ex),df.FacetNormal(self.mesh))
        
        # Resolution of the variationnal problem
        
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        start = time.time()
        a = df.inner(df.grad(u),df.grad(v)) * self.dx + u*v*self.dx
        l = f_expr * v * self.dx + h * v * self.ds
        
        A = df.assemble(a)
        L = df.assemble(l)

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

        uref_Vex = interpolate(u_ex,self.V_ex)
        sol_Vex = interpolate(sol,self.V_ex)
        norme_L2 = (assemble((((uref_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uref_Vex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def corr_add(self, i, phi_tild, phi_tild_inter):
        # nonexactBC=True
        params = self.params[i]
                
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)

        f_tild = f_expr + div(grad(phi_tild)) - phi_tild
        
        # Impose Neumann boundary conditions
        normals = df.FacetNormal(self.mesh)
        h = df.inner(grad(u_ex),normals)
        h_tild = h - df.inner(grad(phi_tild),normals)
        
        # Resolution of the variationnal problem
        
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        start = time.time()

        a = df.inner(df.grad(u),df.grad(v)) * self.dx + u*v*self.dx
        l = f_tild * v * self.dx + h_tild * v * self.ds

        A = df.assemble(a)
        L = df.assemble(l)
        
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

        sol = C_tild + phi_tild_inter

        uref_Vex = interpolate(u_ex,self.V_ex)
        
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+phi_tild.vector()[:]
        
        norme_L2 = (assemble((((uref_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uref_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
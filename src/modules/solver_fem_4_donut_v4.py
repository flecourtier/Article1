## SOLVEUR - Cas test 4 - V4 - Donut

# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

from modules.fenics_expressions import *
from modules.geometry import Donut

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
        
    def fem(self, i, u_ref):
        # boundary = "on_boundary"
        params = self.params[i]
        
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)  

        if cd=="homo":
            g = Constant("0.0")
            
        bc = DirichletBC(self.V, g, "on_boundary")
        
        # plot
        # u = df.Function(self.V)
        # u.vector()[:] = 0.0
        # # u.set_all(0)
        # u.vector()[0] = 1.0
        
        # for mycell in cells(self.mesh):
        #     for myfacet in facets(mycell):
        #         for v in vertices(myfacet):
        #             x,y = v.point().x(),v.point().y()
        #             if boundary([x,y]):
        #                 print(x,y,":",x**2+y**2,v.index())
        #                 u.vector()[v.index()] = 1.0
                                            
        # import matplotlib.pyplot as plt
        # plt.figure()
        # c = df.plot(u)
        # plt.colorbar(c)
        # plt.show()

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        
        start = time.time()

        # a = - div(mat*grad(u)) * v * self.dx
        a = inner(grad(u),grad(v)) * self.dx
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

        uref_Vex = interpolate(u_ref,self.V_ex)
        sol_Vex = interpolate(sol,self.V_ex)
        norme_L2 = (assemble((((uref_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uref_Vex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def corr_add(self, i, phi_tild, u_ref, nonexactBC=True):
        # nonexactBC=True
        params = self.params[i]
                
        f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        
        f_tild = f_expr + div(grad(phi_tild))
        
        if nonexactBC:
            g_tild = Function(self.V)
            phi_tild_inter = project(phi_tild, self.V)
            g_tild.vector()[:] = -(phi_tild_inter.vector()[:])   
        else:
            g_tild = Constant("0.0")     
         
        # def boundary_D(x,on_boundary):
        #     return on_boundary and x[0]**2+x[1]**2<0.75**2 #and x[0]**2+x[1]**2>1.0-DOLFIN_EPS        
        
        # class BoundaryN(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return on_boundary and x[1]<DOLFIN_EPS

        # boundary_N = FacetFunction('size_t', mesh)
        # bcN = BoundaryN()
        # bcN.mark(boundary_N, 0)
        # ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_N)
            
        bc = DirichletBC(self.V, g_tild, "on_boundary")

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

        uref_Vex = interpolate(u_ref,self.V_ex)
        
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+phi_tild.vector()[:]
        
        norme_L2 = (assemble((((uref_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uref_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

# from modfiredrake.firedrake_expressions.firedrake_expressions import *
from testcases.geometry.geometry_2D import Square

from firedrake import *
import firedrake as fd
# import mshr
import time
import numpy as np
from pathlib import Path

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
        
        self.V_ex = FunctionSpace(self.mesh, "CG", self.high_degree)

    def __create_FEM_domain(self):
        nb_vert = self.N+1
        print("Create FEM domain")

        # check if pb_considered is instance of Square class
        if isinstance(self.pb_considered.geometry, Square):
            box = np.array(self.pb_considered.geometry.box)
            Lx, Ly = box[0,1] - box[0,0], box[1,1] - box[1,0]
            origin = (box[0,0], box[1,0])
            
            start = time.time()
            mesh = RectangleMesh(nb_vert - 1, nb_vert - 1, Lx, Ly, originX=origin[0], originY=origin[1])
            end = time.time()
            
            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import axes3d
            # fig, axes = plt.subplots(1, 1)
            # triplot(mesh, axes=axes)
            # axes.legend();
            # plt.show()

            if print_time:
                print("Time to generate mesh: ", end-start)
            self.times_fem["mesh"] = end-start
            self.times_corr_add["mesh"] = end-start
        else:
            raise ValueError("Geometry not implemented")
        
        V = FunctionSpace(mesh, "CG", self.degree)
        dx = Measure("dx", domain=mesh)
        
        # self.h = self.__hmax(mesh)
        # print("hmax = ",self.h)

        return mesh, V, dx
    
    def fem(self, i):
        boundary = "on_boundary"
        
        params = self.params[i]
        
        print("degree of V = ",self.V.ufl_element().degree())
        
        x, y = SpatialCoordinate(self.mesh)
        fct_f = self.pb_considered.f(fd, [x,y], params)
        f_expr = Function(self.V).interpolate(fct_f)
        fct_uex = self.pb_considered.u_ex(fd, [x,y], params)
        u_ex = Function(self.V).interpolate(fct_uex)
        
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        a = inner(grad(u), grad(v))*self.dx
        L = f_expr*v*self.dx

        g = Constant(0.0)
        bc0 = DirichletBC(self.V.sub(0), g, boundary)

        sol = Function(self.V)
        solve(a == L, sol, bcs=bc0)

        # VTKFile("sol.pvd").write(sol)
        
        # compute the relative L2 error

        uex_Vex = Function(self.V).interpolate(u_ex)
        sol_Vex = Function(self.V).interpolate(sol)
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    # def corr_add(self, i, phi_tild, nonexactBC=False):
    #     boundary = "on_boundary"

    #     params = self.params[i]
    #     f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
    #     u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
    #     f_tild = f_expr + div(grad(phi_tild))

    #     if not nonexactBC:
    #         g = Constant(0.0)
    #     else:
    #         g = Function(self.V)
    #         # phi_tild_inter = interpolate(phi_tild, self.V)
    #         phi_tild_inter = project(phi_tild, self.V)
    #         g.vector()[:] = -(phi_tild_inter.vector()[:])        
    #     bc = DirichletBC(self.V, g, boundary)

    #     u = TrialFunction(self.V)
    #     v = TestFunction(self.V)
        
    #     # Resolution of the variationnal problem
    #     start = time.time()
    #     a = inner(grad(u), grad(v)) * self.dx
    #     l = f_tild * v * self.dx

    #     A = df.assemble(a)
    #     L = df.assemble(l)
    #     bc.apply(A, L)

    #     end = time.time()

    #     if print_time:
    #         print("Time to assemble the matrix : ",end-start)
    #     self.times_corr_add["assemble"] = end-start

    #     C_tild = Function(self.V)

    #     start = time.time()
    #     solve(A,C_tild.vector(),L)
    #     end = time.time()

    #     if print_time:
    #         print("Time to solve the system :",end-start)
    #     self.times_corr_add["solve"] = end-start

    #     sol = C_tild + phi_tild

    #     uex_Vex = interpolate(u_ex,self.V_ex)
        
    #     C_Vex = interpolate(C_tild,self.V_ex)
    #     sol_Vex = Function(self.V_ex)
    #     sol_Vex.vector()[:] = (C_Vex.vector()[:])+phi_tild.vector()[:]
        
    #     norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) / (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
    #     return sol,C_tild,norme_L2
    
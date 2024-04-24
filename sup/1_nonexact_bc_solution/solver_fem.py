# homogeneous = True
cd = "homo"
print_time=False

###########
# Imports #
###########

from fenics_expressions import *
from Problem import HomoSolOnUnitSquare

from dolfin import *
import dolfin as df
import time

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["representation"] = "uflacs"
# parameters["form_compiler"]["quadrature_degree"] = 10

#######
# FEM #
#######

class FEMSolver():
    def __init__(self,nb_cell):
        self.N = nb_cell

        self.pb_considered = HomoSolOnUnitSquare()
        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()

    def __create_FEM_domain(self):
        # UnitSquareMesh
        nb_vert = self.N+1
        start = time.time()
        mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), nb_vert - 1, nb_vert - 1)
        end = time.time()

        if print_time:
            print("Time to generate mesh: ", end-start)
        self.times_fem["mesh"] = end-start
        self.times_corr_add["mesh"] = end-start
        
        V = FunctionSpace(mesh, "CG", 1)
        dx = Measure("dx", domain=mesh)
        
        # self.ds = Measure("ds", domain=mesh)

        return mesh, V, dx

    def fem(self, i, get_error=True):
        boundary = "on_boundary"

        f_expr = FExpr(degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        if get_error:
            u_ex = UexExpr(degree=10, domain=self.mesh, pb_considered=self.pb_considered)
            
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

        norme_L2 = None
        if get_error:
            norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def corr_add(self, eps, get_error=True, change_g=False):
        boundary = "on_boundary"

        f_expr = FExpr(degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        u_ex = UexExpr(degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        P = PExpr(degree=10, domain=self.mesh, pb_considered=self.pb_considered)
        phi_tild = u_ex + eps * P

        f_tild = f_expr + div(grad(phi_tild))

        g = Constant(0.0)
        if change_g:
            u_ex_inter = interpolate(u_ex, self.V)
            P_inter = interpolate(P, self.V)
            
            g = Function(self.V)
            g.vector()[:] = -(u_ex_inter.vector()[:] + eps * P_inter.vector()[:])
            
        # check = assemble(u_ex**2 * self.ds)
        # print("u_ex :",check)

        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        start = time.time()
        a = inner(grad(u), grad(v)) * self.dx
        l = f_tild * v * self.dx
        
        # a += 1e10*u*v*self.ds
        # l += 1e10*g*v*self.ds
                  
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
        
        # solve(a==l, C_tild, bcs=bc)

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_corr_add["solve"] = end-start

        sol = C_tild + phi_tild

        norme_L2 = None
        if get_error:
            norme_L2 = (assemble((((u_ex - sol)) ** 2) * self.dx) ** (0.5)) / (assemble((((u_ex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
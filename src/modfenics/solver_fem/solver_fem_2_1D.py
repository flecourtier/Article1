###########
# Imports #
###########

from modfenics.fenics_expressions.fenics_expressions import *
from testcases.geometry.geometry_1D import Line

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
    
    def fem(self, i, plot_result=False, filename=None):
        boundary = "on_boundary"

        params = self.params[i]
        r,Pe = params
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        f_expr = r
        
        g = Constant("0.0")
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = grad(u)[0] * v * self.dx + 1.0/Pe * inner(grad(u), grad(v)) * self.dx
        l = f_expr * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        sol = Function(self.V)
        solve(A,sol.vector(),L)
        
        if plot_result or filename is not None:
            u_ex_inter = interpolate(u_ex,self.V)
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(6,3))
            
            plt.subplot(1,2,1)
            plot(sol,label="sol")
            plot(u_ex_inter,label="u_ex")
            plt.title("solution of FEM")
            plt.legend()
            
            plt.subplot(1,2,2)
            error_sol = Function(self.V)
            error_sol.vector()[:] = abs(sol.vector()[:] - u_ex_inter.vector()[:])
            plot(error_sol)
            plt.title("error on sol")
            
            if plot_result:
                plt.show()
                
            if filename is not None:
                plt.savefig(filename)
                
            plt.close()
            
        uex_Vex = interpolate(u_ex,self.V_ex)
        sol_Vex = interpolate(sol,self.V_ex)
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) #/ (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))

        return sol,norme_L2

    def __plot_result(self, u_ex_inter, phi_tild, C_ex, C_tild, sol, plot_result=False, filename=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,3))
        
        plt.subplot(1,4,1)
        plot(u_ex_inter,label="u_ex",color="orange")
        plot(phi_tild,label="u_theta")
        plt.title("prediction")
        plt.legend()
        
        plt.subplot(1,4,2)
        plot(C_ex,label="C_ex",color="orange")
        plot(C_tild,label="C_tild")
        plt.title("correction")
        plt.legend()
        
        plt.subplot(1,4,3)
        plot(sol,label="sol")
        plot(u_ex_inter,label="u_ex")
        plt.title("solution after correction")
        plt.legend()
        
        plt.subplot(1,4,4)
        error_sol = Function(self.V)
        error_sol.vector()[:] = abs(sol.vector()[:] - u_ex_inter.vector()[:])
        plot(error_sol)
        plt.title("error on sol (corrected)")
        
        if plot_result:
            plt.show()
            
        if filename is not None:
            plt.savefig(filename)
            
        plt.close()
        
    def corr_add(self, i, phi_tild, phi_tild_inter, plot_result=False, filename=None):
        # phi_tild on V_ex; phi_tild_inter on V
        boundary = "on_boundary"

        params = self.params[i]
        r,Pe = params
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        f_expr = r
        f_tild = f_expr - grad(phi_tild)[0] + 1.0/Pe * div(grad(phi_tild))

        g = Constant(0.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = grad(u)[0] * v * self.dx + 1.0/Pe * inner(grad(u), grad(v)) * self.dx
        l = f_tild * v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        bc.apply(A, L)

        C_tild = Function(self.V)
        solve(A,C_tild.vector(),L)
        sol = Function(self.V)
        sol.vector()[:] = C_tild.vector()[:] + phi_tild_inter.vector()[:]
        
        if plot_result or filename is not None:
            u_ex_inter = interpolate(u_ex,self.V)
            C_ex = df.Function(self.V)
            C_ex.vector()[:] = u_ex_inter.vector()[:] - phi_tild_inter.vector()[:]
            self.__plot_result(u_ex_inter,phi_tild,C_ex,C_tild,sol,filename=filename)

        uex_Vex = interpolate(u_ex,self.V_ex)        
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = (C_Vex.vector()[:])+phi_tild.vector()[:]
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) #/ (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2    
    
    def corr_mult(self, i, phi_tild, phi_tild_inter, type="strong", plot_result=False, filename=None):
        # phi_tild on V_ex; phi_tild_inter on V
        assert type in ["strong","weak"]
        boundary = "on_boundary"

        params = self.params[i]
        r,Pe = params
        u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
        f_expr = r

        g = Constant(1.0)
        bc = DirichletBC(self.V, g, boundary)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Resolution of the variationnal problem
        a = grad(phi_tild*u)[0] * phi_tild*v * self.dx + 1.0/Pe * inner(grad(phi_tild*u), grad(phi_tild*v)) * self.dx
        l = f_expr * phi_tild*v * self.dx

        A = df.assemble(a)
        L = df.assemble(l)
        if type == "strong":
            bc.apply(A, L)

        C_tild = Function(self.V)
        solve(A,C_tild.vector(),L)
        sol = Function(self.V)
        sol.vector()[:] = phi_tild_inter.vector()[:] * C_tild.vector()[:]
        
        if plot_result or filename is not None:
            u_ex_inter = interpolate(u_ex,self.V)
            C_ex = df.Function(self.V)
            C_ex.vector()[:] = u_ex_inter.vector()[:]/phi_tild_inter.vector()[:]
            self.__plot_result(u_ex_inter,phi_tild,C_ex,C_tild,sol,filename=filename)
            
        
        uex_Vex = interpolate(u_ex,self.V_ex)
        C_Vex = interpolate(C_tild,self.V_ex)
        sol_Vex = Function(self.V_ex)
        sol_Vex.vector()[:] = C_Vex.vector()[:] * phi_tild.vector()[:]
        sol_Vex = sol_Vex
        norme_L2 = (assemble((((uex_Vex - sol_Vex)) ** 2) * self.dx) ** (0.5)) #/ (assemble((((uex_Vex)) ** 2) * self.dx) ** (0.5))
        
        return sol,C_tild,norme_L2
    
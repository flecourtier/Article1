print_time=False

###########
# Imports #
###########

from modfenicsx.fenicsx_expressions.fenicsx_expressions import *
from testcases.geometry.geometry_2D import Square

from petsc4py.PETSc import ScalarType
import dolfinx as dfx
import dolfinx 
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import time
import numpy as np
import ufl
from pathlib import Path

current = Path(__file__).parent.parent

#######
# FEM #
#######

class FEMSolver():
    def __init__(self,nb_cell,params,problem,degree=1,high_degree=10):
        self.N = nb_cell
        self.params = params
        self.pb_considered = problem
        self.degree = degree
        
        self.times_fem = {}
        self.times_corr_add = {}
        self.mesh,self.V,self.dx = self.__create_FEM_domain()
        self.h = self.get_hmax()
        print("hmax = ",self.h)
        
        # to compute error
        self.high_degree = high_degree 
        self.V_ex = dfx.fem.functionspace(self.mesh, ("CG", self.high_degree))

    def __create_FEM_domain(self):
        nb_vert = self.N+1

        # check if pb_considered is instance of Square class
        if isinstance(self.pb_considered.geometry, Square):
            box = np.array(self.pb_considered.geometry.box)
            start = time.time()
            mesh = dfx.mesh.create_rectangle(
                comm=MPI.COMM_WORLD,
                points=((box[0,0], box[1,0]), (box[0,1], box[1,1])),
                n=(nb_vert-1, nb_vert-1),
                cell_type = dfx.mesh.CellType.triangle,
            )
            end = time.time()
            
            if print_time:
                print("Time to generate mesh: ", end-start)
            self.times_fem["mesh"] = end-start
            self.times_corr_add["mesh"] = end-start
        else:
            raise ValueError("Geometry not implemented")
        
        V = dfx.fem.functionspace(mesh, ("CG", self.degree))
        dx = ufl.Measure("dx", domain=mesh)

        return mesh, V, dx
    
    def get_hmax(self):
        num_cells = (
                self.mesh.topology.index_map(self.mesh.topology.dim).size_local
                + self.mesh.topology.index_map(self.mesh.topology.dim).num_ghosts
            )
        return max(self.mesh.h(2, np.array(list(range(num_cells)))))
    
    def create_bc(self):
        facets = dfx.mesh.locate_entities_boundary(
            self.mesh,
            dim=(self.mesh.topology.dim - 1),
            marker=lambda x: np.full(x.shape[1], True, dtype=bool),
        )

        dofs = dfx.fem.locate_dofs_topological(V=self.V, entity_dim=1, entities=facets)
        bc = dfx.fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=self.V)
        
        return bc
    
    def fem(self, i):
        params = self.params[i]
        
        f_expr = FExpr(params, pb_considered=self.pb_considered)
        f_inter = dfx.fem.Function(self.V_ex)
        f_inter.interpolate(f_expr.eval)
        
        u_ex = UexExpr(params, pb_considered=self.pb_considered)
            
        bc = self.create_bc()

        # Resolution of the variationnal problem
        start = time.time()

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * self.dx
        L = f_inter * v * self.dx        
        
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem["assemble"] = end-start

        start = time.time()
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        sol = problem.solve()
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem["solve"] = end-start

        uex_Vex = dfx.fem.Function(self.V_ex)
        uex_Vex.interpolate(u_ex.eval)

        sol_Vex = dfx.fem.Function(self.V_ex)
        sol_Vex.interpolate(sol)

        # PROJECTION
        # uh_Vex = fem.Function(self.V_ex)
        # u1_2_u2_nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        #     uh_Vex.function_space.mesh,
        #     uh_Vex.function_space.element,
        #     self.V.mesh,
        #     padding=1e-3,
        # )
        # uh_Vex.interpolate(
        #     uh, nmm_interpolation_data=u1_2_u2_nmm_data
        # )
        # uh_Vex.x.scatter_forward()

        # Compute the error in the higher order function space
        e_Vex = dfx.fem.Function(self.V_ex)
        e_Vex.x.array[:] = uex_Vex.x.array - sol_Vex.x.array

        # Integrate the error
        error = dfx.fem.form(ufl.inner(e_Vex,e_Vex) * self.dx)
        error_local = dfx.fem.assemble_scalar(error)
        error_global = self.mesh.comm.allreduce(error_local, op=MPI.SUM)
        
        norm = dfx.fem.form(ufl.inner(uex_Vex, uex_Vex) * self.dx)
        norm_local = dfx.fem.assemble_scalar(norm)
        norm_global = self.mesh.comm.allreduce(norm_local, op=MPI.SUM)
        
        norme_L2 = np.sqrt(error_global/norm_global)
        
        
        return sol,norme_L2
    
    def corr_add(self, i, phi_tild, phi_tild_inter):
        params = self.params[i]
        
        f_expr = FExpr(params, pb_considered=self.pb_considered)
        f_inter = dfx.fem.Function(self.V_ex)
        f_inter.interpolate(f_expr.eval)
        f_tild = f_inter + ufl.div(ufl.grad(phi_tild))
        
        u_ex = UexExpr(params, pb_considered=self.pb_considered)
            
        bc = self.create_bc()

        # Resolution of the variationnal problem
        start = time.time()

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * self.dx
        L = f_tild * v * self.dx        
        
        end = time.time()

        if print_time:
            print("Time to assemble the matrix : ",end-start)
        self.times_fem["assemble"] = end-start

        start = time.time()
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        C_tild = problem.solve()
        sol = C_tild + phi_tild_inter
        end = time.time()

        if print_time:
            print("Time to solve the system :",end-start)
        self.times_fem["solve"] = end-start

        uex_Vex = dfx.fem.Function(self.V_ex)
        uex_Vex.interpolate(u_ex.eval)

        Ctild_Vex = dfx.fem.Function(self.V_ex)
        Ctild_Vex.interpolate(C_tild)
        sol_Vex = dfx.fem.Function(self.V_ex)
        sol_Vex.x.array[:] = Ctild_Vex.x.array + phi_tild.x.array

        # PROJECTION
        # uh_Vex = fem.Function(self.V_ex)
        # u1_2_u2_nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        #     uh_Vex.function_space.mesh,
        #     uh_Vex.function_space.element,
        #     self.V.mesh,
        #     padding=1e-3,
        # )
        # uh_Vex.interpolate(
        #     uh, nmm_interpolation_data=u1_2_u2_nmm_data
        # )
        # uh_Vex.x.scatter_forward()

        # Compute the error in the higher order function space
        e_Vex = dfx.fem.Function(self.V_ex)
        e_Vex.x.array[:] = uex_Vex.x.array - sol_Vex.x.array

        # Integrate the error
        error = dfx.fem.form(ufl.inner(e_Vex,e_Vex) * self.dx)
        error_local = dfx.fem.assemble_scalar(error)
        error_global = self.mesh.comm.allreduce(error_local, op=MPI.SUM)
        
        norm = dfx.fem.form(ufl.inner(uex_Vex, uex_Vex) * self.dx)
        norm_local = dfx.fem.assemble_scalar(norm)
        norm_global = self.mesh.comm.allreduce(norm_local, op=MPI.SUM)
        
        norme_L2 = np.sqrt(error_global/norm_global)
        
        
        return sol,C_tild,norme_L2

    # def corr_add(self, i, phi_tild):
    #     # phi_tild defined on V_ex, error compute on V_ex
    #     boundary = "on_boundary"

    #     params = self.params[i]
    #     f_expr = FExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
    #     u_ex = UexExpr(params, degree=self.high_degree, domain=self.mesh, pb_considered=self.pb_considered)
    #     f_tild = f_expr + div(grad(phi_tild))

    #     g = Constant(0.0)
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
    
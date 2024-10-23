from modfenics.solver_fem.PoissonDirSquareFEMSolver import PoissonDirSquareFEMSolver

def get_solver_type(testcase,version):
    if testcase in [1,2]:
        return PoissonDirSquareFEMSolver
    else:
        pass
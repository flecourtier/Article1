from modfenics.solver_fem.PoissonDirSquareFEMSolver import PoissonDirSquareFEMSolver
from modfenics.solver_fem.EllipticDirSquareFEMSolver import EllipticDirSquareFEMSolver

def get_solver_type(testcase,version):
    if testcase in [1,2]:
        return PoissonDirSquareFEMSolver
    elif testcase == 3:
        return EllipticDirSquareFEMSolver
    else:
        pass
    
# TODO
# def run_reference_solutions():
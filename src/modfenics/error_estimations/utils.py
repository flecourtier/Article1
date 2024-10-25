from modfenics.solver_fem.PoissonDirFEMSolver import PoissonDirSquareFEMSolver,PoissonDirDonutFEMSolver
from modfenics.solver_fem.EllipticDirFEMSolver import EllipticDirSquareFEMSolver
from modfenics.solver_fem.PoissonMixteFEMSolver import PoissonMixteDonutFEMSolver
from modfenics.solver_fem.PoissonModNeuFEMSolver import PoissonModNeuDonutFEMSolver

def get_solver_type(testcase,version):
    if testcase in [1,2]:
        return PoissonDirSquareFEMSolver
    elif testcase == 3:
        return EllipticDirSquareFEMSolver
    elif testcase == 4:
        return PoissonDirDonutFEMSolver
    elif testcase == 5:
        return PoissonMixteDonutFEMSolver
    elif testcase == 6:
        return PoissonModNeuDonutFEMSolver
    else:
        pass
    
# TODO
# def run_reference_solutions():
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Extract testcase, version, and param_num from arguments.")

# Add arguments
parser.add_argument('--version', type=str, required=True, help="Version of the testcase")
parser.add_argument('--param_num', type=int, required=True, help="Number of parameters")
parser.add_argument('--largenet', action='store_true', help="Largenet for medium testcase")

# Parse the arguments
args = parser.parse_args()

# Access the values
testcase = 3
size_param = args.version
param_num = args.param_num
largenet = args.largenet
high_degree = 10

# Print the extracted values
print("#############################\n#############################")

print(f"Testcase: {testcase}")
print(f"Version: {size_param}")
print(f"Parameter Number: {param_num}")
print(f"Largenet: {largenet}")

assert testcase == 3
assert size_param in ["small","medium","big","new"]

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import dolfin as df
import seaborn as sns
import dataframe_image as dfi

# from modfenics.fenics_expressions.fenics_expressions_2D import UexExpr
from scimba.equations.domain import SpaceTensor
from scimba.equations import domain

from testcases.utils import create_tree
from testcases.geometry.geometry_2D import Square
from modfenics.fenics_expressions.fenics_expressions import FExpr,AnisotropyExpr
from modfenics.utils import get_param,compute_slope
from modfenics.error_estimations.fem import compute_error_estimations_fem_deg,compute_error_estimations_fem_all
from modfenics.error_estimations.add import compute_error_estimations_Corr_deg,compute_error_estimations_Corr_all,plot_Corr_vs_FEM
from modfenics.error_estimations.mult import compute_error_estimations_Mult_deg,compute_error_estimations_Mult_all,plot_Mult_vs_FEM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if size_param != "new":
    from testcases.networks.test_2D.test_3.test_3_v1 import Run_laplacian2D,Poisson_2D
else:
    from testcases.networks.test_2D.test_3.test_3_v2 import Run_laplacian2D,Poisson_2D
from testcases.problem.problem_2D import TestCase3
from modfenics.solver_fem.EllipticDirSquareFEMSolver import EllipticDirSquareFEMSolver

problem = TestCase3(version=size_param)

dim_params = problem.nb_parameters
result_dir = "../../../results/fenics/test_2D/testcase"+str(testcase)+"/"+size_param
if size_param == "medium" and largenet:
    result_dir += "_largenet"
result_dir += "/cvg/"
create_tree(result_dir)

if size_param != "new":
    pde = Poisson_2D(size_param)
    trainer,u_theta = Run_laplacian2D(pde,size_param,largenet)
else:
    pde = Poisson_2D()
    trainer,u_theta = Run_laplacian2D(pde)

compute_error_estimations_fem_all(param_num,problem,high_degree,new_run=False,result_dir=result_dir,plot_cvg=True)
compute_error_estimations_Corr_all(param_num,problem,high_degree,u_theta,new_run=False,result_dir=result_dir,plot_cvg=True)
plot_Corr_vs_FEM(param_num,problem,result_dir=result_dir)
tab_M = [0.0,0.1,1.0,100.0]
for M in tab_M:
    print("#### M = ",M)
    compute_error_estimations_Mult_all(param_num,problem,high_degree,u_theta,M=M,new_run=False,result_dir=result_dir,plot_cvg=True)
plot_Mult_vs_FEM(param_num,problem,result_dir=result_dir)



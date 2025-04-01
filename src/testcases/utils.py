import os
import numpy as np
import matplotlib.pyplot as plt

def create_tree(path):
    path_split = path.split("/")
    if path[0]=="/":
        path_split = path_split[1:]
        start = "/"
    else:
        start = ""
    for i in range(1,len(path_split)+1):
        subdir = "/".join(path_split[:i])
        if not os.path.isdir(start+subdir):
            os.mkdir(start+subdir)
            
def get_random_param(i,parameter_domain):
    # pick 1 random parameter at a given index
    dim_params = len(parameter_domain)
    np.random.seed(0)
    for j in range(i):
        param = []
        for k in range(dim_params):
            param.append(np.random.uniform(parameter_domain[k][0], parameter_domain[k][1]))
    param = np.round(param, 2)
    return param

def get_random_params(n_params,parameter_domain):
    # pick n random parameters
    dim_params = len(parameter_domain)

    np.random.seed(0)
    params = []
    for i in range(dim_params):
        parami = np.random.uniform(parameter_domain[i][0], parameter_domain[i][1], n_params)
        params.append(parami)
    params = np.array(params).T
    return params

def select_param(problem,param_num):
    try:
        set_params = problem.set_params
    except:
        set_params = None
        
    if set_params is not None:
        assert 1 <= param_num <= len(set_params), f"param_num={param_num} should be less than {len(set_params)}"
        param = set_params[param_num-1]
    else:
        parameter_domain = problem.parameter_domain
        param = get_random_param(param_num,parameter_domain)
        if problem.testcase == 3 and param_num == 3:
            print("ATTENTION : paramètre 3 choisi à la main")
            param = [0.46,0.52,0.12,0.05]
        
    return param

def compute_slope(i,tab_nb_vert,tab_err):
    start = [tab_nb_vert[i],tab_err[i]]
    end = [tab_nb_vert[i-1],tab_err[i-1]]
    third = [end[0],start[1]]

    tri_x = [end[0], third[0], start[0], end[0]]
    tri_y = [end[1], third[1], start[1], end[1]
    ]
    plt.plot(tri_x, tri_y, "k--", linewidth=0.5)

    slope = -(np.log(start[1])-np.log(end[1]))/(np.log(start[0])-np.log(end[0]))
    slope = slope.round(2)
    
    vert_mid = [(end[0]+third[0])/2., (end[1]+third[1])/2.]
    
    return slope,vert_mid
import json
import numpy as np

def choice_dimension():
    return [1, 2, 3]

def choice_testcase(dimension):
    if dimension == 1:
        choice = np.arange(1,2+1)
        pb = ["Poisson + Dirichlet","Poisson + Elliptic and Convection Dominate Regime"]
    elif dimension == 2:
        choice = np.arange(1,6+1)
        pb = ["Carré + Poisson + Dirichlet (basse fréquence)","Carré + Poisson + Dirichlet (haute fréquence)","Carré + Elliptique + Dirichlet (non analytique)","Donut + Poisson + Dirichlet (analytique)","Donut + Poisson + Mixte (analytique)","Donut + Poisson modifié + Neumann (analytique)"]
    else:
        choice = np.arange(1,1+1)
        pb = ["Cube + Poisson + Dirichlet (basse fréquence)"]
    assert len(pb) == len(choice), "The number of problems must be equal to the number of choices"
    return list(choice),pb

def get_str(pb):
    infos = ""
    for i in range(len(pb)):
        infos += f"# TestCase {i+1} : {pb[i]}"
        if i != len(pb)-1:
            infos += "\n"
    return infos

def choice_version(dimension,testcase):
    if dimension == 1:
        if testcase == 1:
            return [1,2]
        else:
            return [1,2]
    elif dimension == 2:
        if testcase == 1:
            return [1,2]
        elif testcase == 2:
            return [1]
        elif testcase == 3:
            return ["big","medium","medium_largenet","small","new"]
        elif testcase == 4:
            return [1]
        elif testcase == 5:
            return [1,2,3]
        elif testcase == 6:
            return [1,2]
    else:
        if testcase == 1:
            return [1]
        
def check_config(config):
    # Check if the config file is valid
    assert "dimension" in config, "dimension is missing in the config file"
    assert "testcase" in config, "testcase is missing in the config file"
    assert "version" in config, "version is missing in the config file"
    # assert "param_num" in config, "param_num is missing in the config file"
    
    assert config["dimension"] in choice_dimension(), "dimension is invalid"
    assert config["testcase"] in choice_testcase(config["dimension"])[0], "testcase is invalid"
    assert config["version"] in choice_version(config["dimension"], config["testcase"]), "version is invalid"
    
    print("Config file is valid")
    
def print_config(config):
    # Print the config file
    for key, value in config.items():
        print(f"# {key} : {value}")

def ask_user():
    print("### Create the configuration file ###")
    
    # Question 1: Ask for the dimension
    possibility = choice_dimension()
    print(f"## Dimension \n The available dimensions are: {possibility}")
    dimension = int(input(f"-> Which dimension would you like to consider? "))
    assert dimension in [1, 2, 3], f"Dimension must be : {possibility}"
    print(f"Dimension: {dimension}")
    
    # Question 2: Ask for the test case
    possibility,pb = choice_testcase(dimension)
    print(f"## Testcase \n The available testcases are: {possibility}")
    print(get_str(pb))
    test_case = int(input(f"-> Which testcase would you like to run? "))
    assert test_case in possibility, f"Testcase must be : {possibility}"

    # Question 3: Ask for the version
    possibility = choice_version(dimension,test_case)
    print(f"## Version \n The available versions are: {possibility}")
    version = input("-> Which version would you like to choose? ")
    if dimension == 2 and test_case != 3 or dimension != 2:
        version = int(version)
    assert version in possibility, f"Version must be : {possibility}"

    # Question 4: Ask for the parameter number
    print("## Parameter Number")
    param_number = int(input("-> Which parameter number would you like to consider? "))
    
    # Store the data in a dictionary
    config = {
        "dimension" : dimension,
        "testcase": test_case,
        "version": version,
        "param_num": param_number
    }

    # Write the dictionary to a config.json file
    with open("config.json", "w") as json_file:
        json.dump(config, json_file, indent=4)

    print("### Configuration has been saved to config.json ###")
    print_config(config)
    
def read_config(configfile="config.json"):
    # Read the config file
    with open(configfile, "r") as json_file:
        config = json.load(json_file)

    # Check the config file
    print("### Configuration has been read from config.json ###")
    print_config(config)

    return config

# Call the function
# ask_user()

import numpy as np
import os
import sys
import shutil
from pathlib import Path
from modules.utils import create_tree

current = Path(__file__).parent.parent

antora_dir = current / "docs" / "antora" / "modules" / "ROOT" / "assets" / "images"

dest_dir = ["training", "cvg", "facteurs"]
for dest in dest_dir:
    create_tree(str(antora_dir / dest))

networks_file = current / "networks"
results_file = current / "results"

testcases = [1,2,3]

def copy_training(testcase):
    tocopy = networks_file / ("test_fe"+str(testcase)+".png")
    dest = antora_dir / "training" / ("test_fe"+str(testcase)+".png")
    
    shutil.copy(tocopy, dest)
    
def copy_cvg(testcase):
    tocopy = results_file / ("testcase"+str(testcase)) / "cvg"
    dest = antora_dir / "cvg"
    print(tocopy)
    if os.path.exists(tocopy):
        for file in os.listdir(tocopy):
            if file.endswith(".png"):
                file_tocopy = tocopy / file
                file_dest = dest / file
                shutil.copy(file_tocopy, file_dest)
                
def copy_facteurs(testcase):
    tocopy = results_file / ("testcase"+str(testcase)) / "facteurs"
    dest = antora_dir / "facteurs"
    print(tocopy)
    if os.path.exists(tocopy):
        for file in os.listdir(tocopy):
            if file.endswith(".png"):
                file_tocopy = tocopy / file
                file_dest = dest / file
                shutil.copy(file_tocopy, file_dest)
        
for testcase in testcases:
    copy_training(testcase)
    copy_cvg(testcase)
    copy_facteurs(testcase)
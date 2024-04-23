import os

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
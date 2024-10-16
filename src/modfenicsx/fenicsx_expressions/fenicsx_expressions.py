###########
# Imports #
###########

# import dolfinx as dolx
import numpy as np

######################
# FENICSX Expressions #
######################

class FExpr:
    def __init__(self, params, pb_considered):
        self.mu = params
        self.pb_considered = pb_considered

    def eval(self, xyz):
        x,y,z = xyz        
        return self.pb_considered.f(np, [x,y], self.mu)
    
class UexExpr:
    def __init__(self, params, pb_considered):
        self.mu = params
        self.pb_considered = pb_considered

    def eval(self, xyz):
        x,y,z = xyz        
        return self.pb_considered.u_ex(np, [x,y], self.mu)
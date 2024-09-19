from modules.geometry_1D import Line1
from math import *
import dolfin

class TestCase1:
    def __init__(self):
        self.geometry = Line1() 
        self.nb_parameters = 3
        self.parameter_domain = [[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]

    def u_ex(self, pre, xy, mu):
        if pre is dolfin:
            x=xy[0]
        else:
            x=xy
        alpha,beta,gamma = mu
        return alpha*pre.sin(2.0*pre.pi*x) + beta*pre.sin(4.0*pre.pi*x) + gamma*pre.sin(6.0*pre.pi*x)

    def du_ex_dx(self, pre, xy, mu):
        if pre is dolfin:
            x=xy[0]
        else:
            x=xy
        alpha,beta,gamma = mu
        return 2.0*pre.pi*alpha*pre.cos(2.0*pre.pi*x) + 4.0*pre.pi*beta*pre.cos(4.0*pre.pi*x) + 6.0*pre.pi*gamma*pre.cos(6.0*pre.pi*x)
    
    def d2u_ex_dx2(self, pre, xy, mu):
        return -self.f(pre, xy, mu)

    def f(self, pre, xy, mu):
        if pre is dolfin:
            x=xy[0]
        else:
            x=xy
        alpha,beta,gamma = mu
        return pre.pi**2 * (4.0*alpha*pre.sin(2.0*pre.pi*x) + 16.0*beta*pre.sin(4.0*pre.pi*x) + 36.0*gamma*pre.sin(6.0*pre.pi*x))

    def g(self, pre, xy, mu):
        return 0.0
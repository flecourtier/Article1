from testcases.geometry.geometry_1D import Line1
from math import *
import dolfin
import abc
class TestCase1D(abc.ABC):
    def __init__(self,testcase,version):
        self.testcase = testcase
        self.version = version
        self.dim = 1
        
        @property
        @abc.abstractmethod
        def geometry(self):
            pass
        @property
        @abc.abstractmethod
        def nb_parameters(self):
            pass
        @property
        @abc.abstractmethod
        def parameter_domain(self):
            pass
        @property
        @abc.abstractmethod
        def ana_sol(self):
            pass

class TestCase1(TestCase1D):
    def __init__(self,version=1):
        super().__init__(1,version)
        self.geometry = Line1() 
        self.nb_parameters = 3
        self.parameter_domain = [[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]
        self.version = version
        self.ana_sol = True
        self.set_params = [[0.3, 0.2, 0.1], [0.8, 0.5, 0.8]]
        
    def u_ex(self, pre, xy, mu):
        # if pre is dolfin:
        #     x=xy[0]
        # else:
        x=xy
        alpha,beta,gamma = mu
        return alpha*pre.sin(2.0*pre.pi*x) + beta*pre.sin(4.0*pre.pi*x) + gamma*pre.sin(6.0*pre.pi*x)

    def du_ex_dx(self, pre, xy, mu):
        # if pre is dolfin:
        #     x=xy[0]
        # else:
        x=xy
        alpha,beta,gamma = mu
        return 2.0*pre.pi*alpha*pre.cos(2.0*pre.pi*x) + 4.0*pre.pi*beta*pre.cos(4.0*pre.pi*x) + 6.0*pre.pi*gamma*pre.cos(6.0*pre.pi*x)
    
    def d2u_ex_dx2(self, pre, xy, mu):
        return -self.f(pre, xy, mu)

    def f(self, pre, xy, mu):
        # if pre is dolfin:
        #     x=xy[0]
        # else:
        x=xy
        alpha,beta,gamma = mu
        return pre.pi**2 * (4.0*alpha*pre.sin(2.0*pre.pi*x) + 16.0*beta*pre.sin(4.0*pre.pi*x) + 36.0*gamma*pre.sin(6.0*pre.pi*x))

    def g(self, pre, xy, mu):
        return 0.0
    
class TestCase2(TestCase1D):
    def __init__(self,version=1):
        super().__init__(2,version)
        self.geometry = Line1() 
        self.nb_parameters = 2
        self.parameter_domain = [[1.0, 2.0],[10.0, 100.0]]
        self.ana_sol = True
        self.set_params = [[1.2,40.0],[1.5,90.0]]

    def u_ex(self, pre, xy, mu):
        # if pre is dolfin:
        #     x=xy[0]
        # else:
        x=xy
        r,Pe = mu
        return r * (x - (pre.exp(Pe*x)-1.0)/(pre.exp(Pe)-1.0) )
    
    def du_ex_dx(self, pre, xy, mu):
        # if pre is dolfin:
        #     x=xy[0]
        # else:
        x=xy
        r,Pe = mu
        return r * (1.0 - Pe*pre.exp(Pe*x)/(pre.exp(Pe)-1.0))
        
    def d2u_ex_dx2(self, pre, xy, mu):
        x=xy
        r,Pe = mu
        return -r * Pe**2 * pre.exp(Pe*x)/(pre.exp(Pe)-1.0)      

    def g(self, pre, xy, mu):
        return 0.0
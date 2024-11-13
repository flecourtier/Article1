from testcases.geometry.geometry_2D import Square1, UnitSquare, UnitCircle, Donut1, Donut2, SquareDonut1
from math import *
import torch
import abc
import sympy

class TestCase2D(abc.ABC):
    def __init__(self,testcase,version):
        self.testcase = testcase
        self.version = version
        self.dim = 2
        
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
        
class TestCase1(TestCase2D):
    def __init__(self,version=1):
        super().__init__(1,version)
        self.geometry = Square1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]
        self.ana_sol = True
        
    def u_ex(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
        ex = pre.exp(-((x-mu1)**2.0 +(y-mu2)**2.0)/2)
        return ex * pre.sin(2*x) * pre.sin(2*y)

    def f(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
        return -pre.exp(-((x - mu1)**2 + (y - mu2)**2)/2) * (((x**2 - 2*mu1*x + mu1**2 - 5)*pre.sin(2*x) + (4*mu1 - 4*x)*pre.cos(2*x)) * pre.sin(2*y) + pre.sin(2*x) * ((y**2 - 2*mu2*y + mu2**2 - 5)*pre.sin(2*y) + (4*mu2 - 4*y)*pre.cos(2*y)))

    def g(self, pre, xy, mu):
        return 0.0
    
class TestCase2(TestCase2D):
    def __init__(self,version=1):
        super().__init__(2,version)
        self.geometry = Square1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]
        self.ana_sol = True
        
    def u_ex(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
        ex = pre.exp(-((x-mu1)**2 +(y-mu2)**2)/2.0)
        return ex * pre.sin(8*x) * pre.sin(8*y)

    def f(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu

        return (16.0*(x-mu1)*pre.sin(8*y)*pre.cos(8*x) - 1.0*(x-mu1)**2.0*pre.sin(8*x)*pre.sin(8*y) + 16.0*(y-mu2)*pre.sin(8*x)*pre.cos(8*y) - 1.0*(y-mu2)**2.0*pre.sin(8*x)*pre.sin(8*y) + 130.0*pre.sin(8*x)*pre.sin(8*y))*pre.exp(-(x-mu1)**2.0/2 - (y-mu2)**2.0/2)

    def g(self, pre, xy, mu):
        return 0.0
    
class TestCase3(TestCase2D):
    def __init__(self,version="big"):
        assert version in ["big","small","medium","new"]
        super().__init__(3,version)
        self.geometry = UnitSquare() 
        self.nb_parameters = 4
        if self.version == "big":
            self.parameter_domain = [[0.4, 0.6],[0.4, 0.6],[0.1, 0.8],[0.01, 1.0]] #c1,c2,sigma,eps
        elif self.version == "small":
            self.parameter_domain = [[0.45, 0.55],[0.45, 0.55],[0.4, 0.6],[0.05, 0.2]] #c1,c2,sigma,eps
        elif self.version == "medium":
            self.parameter_domain = [[0.4, 0.6],[0.4, 0.6],[0.3, 0.6],[0.04, 0.25]] #c1,c2,sigma,eps
        else:
            self.parameter_domain = [[0.4, 0.6],[0.4, 0.6],[0.4, 0.8],[0.5, 1.0]] #c1,c2,sigma,eps
        self.ana_sol = False
        
    def u_ex(self, pre, xy, mu):
        pass
    
    def anisotropy_matrix(self, pre, xy, mu):
        x,y = xy
        _,_,_, eps = mu

        a11 = eps * x**2 + y**2
        a12 = (eps - 1) * x * y
        a21 = (eps - 1) * x * y
        a22 = x**2 + eps * y**2

        return a11, a12, a21, a22

    def f(self, pre, xy, mu):
        x,y=xy
        c1,c2,sigma,eps = mu
        if self.version != "new":
            return pre.exp(-((x - c1) ** 2 + (y - c2) ** 2) / (0.025 * sigma**2))
        else:
            print("f is considered for new version")
            return 10 * pre.exp(-((x - c1) ** 2 + (y - c2) ** 2) / (0.025 * sigma**2))
        
    def g(self, pre, xy, mu):
        return 0.0
        
class TestCase4(TestCase2D):
    def __init__(self,version=1):
        assert version == 1
        super().__init__(4,version)
        self.geometry = Donut1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]
        self.ana_sol = True
   
    def u_ex(self, pre, xy, mu):
        x,y = xy
        mu1,mu2 = mu
        return 1.0/(2*pre.pi)*pre.exp(-1.0/2.0*((x-mu1)**2+(y-mu2)**2))*pre.sin(-1.0/4.0 * (x**2 + y**2 - 1.0))

    def f(self, pre, xy, mu):
        x,y = xy
        mu1,mu2 = mu
        return -0.5*(0.25*x**2*pre.sin(0.25*(x**2 + y**2 - 1)) - x*(mu1 - x)*pre.cos(0.25*(x**2 + y**2 - 1)) - ((mu1 - x)**2 - 1)*pre.sin(0.25*(x**2 + y**2 - 1)) - 0.5*pre.cos(0.25*(x**2 + y**2 - 1)))*pre.exp(-0.5*(-mu1 + x)**2 - 0.5*(-mu2 + y)**2)/pre.pi - 0.5*(0.25*y**2*pre.sin(0.25*(x**2 + y**2 - 1)) - y*(mu2 - y)*pre.cos(0.25*(x**2 + y**2 - 1)) - ((mu2 - y)**2 - 1)*pre.sin(0.25*(x**2 + y**2 - 1)) - 0.5*pre.cos(0.25*(x**2 + y**2 - 1)))*pre.exp(-0.5*(-mu1 + x)**2 - 0.5*(-mu2 + y)**2)/pre.pi
    
    def g(self, pre, xy, mu):
        return self.u_ex(pre, xy, mu)      
        

class TestCase5(TestCase2D):
    def __init__(self,version=1):
        assert version in [1,2,3]
        super().__init__(5,version)
        self.geometry = Donut2()
        self.nb_parameters = 1
        self.parameter_domain = [[0.50000, 0.500001]]
        self.ana_sol = True

    def u_ex(self, pre, xy, mu):
        x,y = xy
        if pre is torch:
            ln = pre.log
        else:
            ln = pre.ln
        return 1.0 - ln(pre.sqrt(x**2 + y**2))/log(4.0)
    
    def grad_uex(self, pre, xy, mu):
        x,y = xy
        coeff = -1.0/log(4.0)
        s = x**2 + y**2
        return coeff*x/s, coeff*y/s
    
    def grad2_uex(self, pre, xy, mu):
        x,y = xy
        coeff = -1.0/log(4.0)
        s = x**2 + y**2
        return coeff * (s - 2*x**2)/s**2, coeff * (s - 2*y**2)/s**2
    
    def grad3_uex(self, pre, xy, mu):
        x,y = xy
        coeff = -1.0/log(4.0)
        s = x**2 + y**2
        return coeff * 2 * x * (x**2 - 3 * y **2)/s**3, coeff * 2 * y * (y**2 - 3 * x **2)/s**3
    
    def f(self, pre, xy, mu):
        x,y = xy
        return 0.0
    
    def h_int(self, pre, xy, mu): # robin
        assert self.version in [1,2,3]
        return 4.0/log(4.0) + 2.0
    
    def h_ext(self, pre, xy, mu): # dirichlet
        return 1.0
    
    def gr(self, pre, xy, mu): # robin
        return self.h_int(pre, xy, mu)
    
    def g(self, pre, xy, mu): # dirichlet
        return self.h_ext(pre, xy, mu)  
    
    
    
class TestCase6(TestCase2D):
    def __init__(self,version=1):
        assert version in [1,2]
        super().__init__(6,version)
        self.geometry = Donut1()
        self.nb_parameters = 1
        self.parameter_domain = [[0.50000, 0.500001]]
        self.ana_sol = True

    def u_ex(self, pre, xy, mu):
        x,y = xy
        return pre.sin(x**2 + y**2)
    
    def f(self, pre, xy, mu):
        x,y = xy
        return (4.0 * (x**2 + y**2) + 1) * pre.sin(x**2 + y**2) - 4.0 * pre.cos(x**2 + y**2)
    
    # def grad_f(self, pre, xy, mu):
    #     x,y = xy
    #     df_dx =  x*((8.0*x**2 + 8.0*y**2 + 2)*pre.cos(x**2 + y**2) + 16.0*pre.sin(x**2 + y**2))
    #     df_dy =  y*((8.0*x**2 + 8.0*y**2 + 2)*pre.cos(x**2 + y**2) + 16.0*pre.sin(x**2 + y**2))
    #     return df_dx, df_dy 
        
    def h_int(self, pre, xy, mu):
        return -cos(1.0/4.0)
    
    def h_ext(self, pre, xy, mu):
        return 2 * cos(1.0)
    
# class TestCase7:
#     def __init__(self,v=1):
#         self.version = v 
#         assert self.version == 1
#         self.geometry = SquareDonut1()
#         self.nb_parameters = 1
#         self.parameter_domain = [[0.50000, 0.500001]]

#     def u_ex(self, pre, xy, mu):
#         x,y = xy
#         PI = pre.pi
#         return pre.sin(2*PI*x)*pre.sin(2*PI*y)
    
#     def grad_uex(self, pre, xy, mu):
#         pass
    
#     def grad2_uex(self, pre, xy, mu):
#         pass
    
#     def grad3_uex(self, pre, xy, mu):
#         pass
    
#     def f(self, pre, xy, mu):
#         x,y = xy
#         PI = pre.pi
#         return (1.0 + 8.0*PI**2)*pre.sin(2*PI*x)*pre.sin(2*PI*y)
    
#     def grad_f(self, pre, xy, mu):
#         pass
        
#     def h_int(self, pre, xy, mu):
#         x,y = xy
#         PI = pre.pi
#         return 2*( y*2*PI*pre.sin(2*PI*x) + x*2*PI*pre.sin(2*PI*y) )
    
#     def h_ext(self, pre, xy, mu):
#         x,y = xy
#         PI = pre.pi
#         return y*2*PI*pre.sin(2*PI*x) + x*2*PI*pre.sin(2*PI*y)
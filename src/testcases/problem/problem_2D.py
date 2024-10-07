from testcases.geometry.geometry_2D import Square1, UnitSquare, UnitCircle, Donut1, Donut2
from math import *
import dolfin

class TestCase1:
    def __init__(self):
        self.geometry = Square1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]

    def u_ex(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
        ex = pre.exp(-((x-mu1)**2.0 +(y-mu2)**2.0)/2)
        return ex * pre.sin(2*x) * pre.sin(2 *y)

    # def u_ex_prime(self, pre, xy, mu):
    #     x,y=xy
    #     du_dx = 
    #     du_dy = 
    #     return du_dx,du_dy

    # def u_ex_prime2(self, pre, xy, mu):
    #     x,y=xy
    #     du_dxx = 
    #     du_dyy = 
    #     return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
        return -pre.exp(-((x - mu1)**2 + (y - mu2)**2)/2) * (((x**2 - 2*mu1*x + mu1**2 - 5)*pre.sin(2*x) + (4*mu1 - 4*x)*pre.cos(2*x)) * pre.sin(2*y) + pre.sin(2*x) * ((y**2 - 2*mu2*y + mu2**2 - 5)*pre.sin(2*y) + (4*mu2 - 4*y)*pre.cos(2*y)))

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0.0
    
class TestCase2:
    def __init__(self):
        self.geometry = Square1() 
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]

    def u_ex(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
        ex = pre.exp(-((x-mu1)**2 +(y-mu2)**2)/2.0)
        return ex * pre.sin(8*x) * pre.sin(8*y)

    # def u_ex_prime(self, pre, xy, mu):
    #     x,y=xy
    #     du_dx = 
    #     du_dy = 
    #     return du_dx,du_dy

    # def u_ex_prime2(self, pre, xy, mu):
    #     x,y=xy
    #     du_dxx = 
    #     du_dyy = 
    #     return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        x,y=xy
        mu1,mu2 = mu
    
        # return pre.exp(-((x-mu1)**2.0 - (y-mu2)**2.0)/2.0) * (16.0*(x-mu1)*pre.sin(8*y)*pre.cos(8*x) - 1.0*(x-mu1)**2.0*pre.sin(8*x)*pre.sin(8*y) + 16.0*(y-mu2)*pre.sin(8*x)*pre.cos(8*y) - 1.0*(y-mu2)**2.0*pre.sin(8*x)*pre.sin(8*y) + 130.0*pre.sin(8*x)*pre.sin(8*y))
        return (16.0*(x-mu1)*pre.sin(8*y)*pre.cos(8*x) - 1.0*(x-mu1)**2.0*pre.sin(8*x)*pre.sin(8*y) + 16.0*(y-mu2)*pre.sin(8*x)*pre.cos(8*y) - 1.0*(y-mu2)**2.0*pre.sin(8*x)*pre.sin(8*y) + 130.0*pre.sin(8*x)*pre.sin(8*y))*pre.exp(-(x-mu1)**2.0/2 - (y-mu2)**2.0/2)

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0.0
    
class TestCase3:
    def __init__(self):
        self.geometry = UnitSquare() 
        self.nb_parameters = 4
        # self.parameter_domain = [
        #         [0.4, 0.40001],  # 0.4 < c1 < 0.6
        #         [0.6, 0.60001],  # 0.4 < c2 < 0.6
        #         [0.8, 0.8001],  # 0.1 < sigma < 0.8
        #         [0.025, 0.025001],  # 0.01 < eps < 1
        #     ]
        self.parameter_domain = [
                [0.4, 0.6],  # 0.4 < c1 < 0.6
                [0.4, 0.6],  # 0.4 < c2 < 0.6
                [0.1, 0.8],  # 0.1 < sigma < 0.8
                [0.01, 1.0],  # 0.01 < eps < 1
            ]

    def u_ex(self, pre, xy, mu):
        pass

    # def u_ex_prime(self, pre, xy, mu):
    #     x,y=xy
    #     du_dx = 
    #     du_dy = 
    #     return du_dx,du_dy

    # def u_ex_prime2(self, pre, xy, mu):
    #     x,y=xy
    #     du_dxx = 
    #     du_dyy = 
    #     return du_dxx,du_dyy
    
    def anisotropy_matrix(self, pre, xy, mu):
        x,y = xy
        c1, c2, sigma, eps = mu

        a11 = eps * x**2 + y**2
        a12 = (eps - 1) * x * y
        a21 = (eps - 1) * x * y
        a22 = x**2 + eps * y**2

        return a11, a12, a21, a22

    def f(self, pre, xy, mu):
        x,y=xy
        c1,c2,sigma,eps = mu
    
        return pre.exp(-((x - c1) ** 2 + (y - c2) ** 2) / (0.025 * sigma**2))
        
    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0.0
    
class TestCase3_new:
    def __init__(self):
        self.geometry = UnitSquare() 
        self.nb_parameters = 4
        self.parameter_domain = [
                [0.4, 0.6],  # 0.4 < c1 < 0.6
                [0.4, 0.6],  # 0.4 < c2 < 0.6
                [0.4, 0.8],  # 0.1 < sigma < 0.8
                [0.5, 1.0],  # 0.5 < eps < 1
            ]

    def u_ex(self, pre, xy, mu):
        pass
    
    def anisotropy_matrix(self, pre, xy, mu):
        x,y = xy
        c1, c2, sigma, eps = mu

        a11 = eps * x**2 + y**2
        a12 = (eps - 1) * x * y
        a21 = (eps - 1) * x * y
        a22 = x**2 + eps * y**2

        return a11, a12, a21, a22

    def f(self, pre, xy, mu):
        x,y=xy
        c1,c2,sigma,eps = mu
    
        return 10 * pre.exp(-((x - c1) ** 2 + (y - c2) ** 2) / (0.025 * sigma**2))
        
    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0.0
    
class TestCase3_small_param(TestCase3):
    def __init__(self):
        super().__init__()
        self.parameter_domain = [
                [0.45, 0.55],  # 0.4 < c1 < 0.6
                [0.45, 0.55],  # 0.4 < c2 < 0.6
                [0.4, 0.6],  # 0.1 < sigma < 0.8
                [0.05, 0.2],  # 0.01 < eps < 1
            ]
        
class TestCase3_medium_param(TestCase3):
    def __init__(self):
        super().__init__()
        self.parameter_domain = [
                [0.4, 0.6],  # 0.4 < c1 < 0.6
                [0.4, 0.6],  # 0.4 < c2 < 0.6
                [0.3, 0.6],  # 0.1 < sigma < 0.8
                [0.04, 0.25],  # 0.01 < eps < 1
            ]
        
        
class TestCase4:
    def __init__(self,v=1):
        self.geometry = Donut1() 
        self.version = v 
        assert self.version == 1
        self.nb_parameters = 2
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001]]

    def u_ex(self, pre, xy, mu):
        x,y = xy
        mu1,mu2 = mu
        return 1.0/(2*pre.pi)*pre.exp(-1.0/2.0*((x-mu1)**2+(y-mu2)**2))*pre.sin(-1.0/4.0 * (x**2 + y**2 - 1.0))

    def f(self, pre, xy, mu):
        x,y = xy
        mu1,mu2 = mu
        return -0.5*(0.25*x**2*pre.sin(0.25*(x**2 + y**2 - 1)) - x*(mu1 - x)*pre.cos(0.25*(x**2 + y**2 - 1)) - ((mu1 - x)**2 - 1)*pre.sin(0.25*(x**2 + y**2 - 1)) - 0.5*pre.cos(0.25*(x**2 + y**2 - 1)))*pre.exp(-0.5*(-mu1 + x)**2 - 0.5*(-mu2 + y)**2)/pre.pi - 0.5*(0.25*y**2*pre.sin(0.25*(x**2 + y**2 - 1)) - y*(mu2 - y)*pre.cos(0.25*(x**2 + y**2 - 1)) - ((mu2 - y)**2 - 1)*pre.sin(0.25*(x**2 + y**2 - 1)) - 0.5*pre.cos(0.25*(x**2 + y**2 - 1)))*pre.exp(-0.5*(-mu1 + x)**2 - 0.5*(-mu2 + y)**2)/pre.pi

    # Dirichlet BC        
    def g(self, pre, xy, mu):
        return self.u_ex(pre, xy, mu)      
        

class TestCase5:
    def __init__(self,v=1):
        self.version = v 
        assert self.version in [1,2,3]
        self.geometry = Donut2()
        self.nb_parameters = 1
        self.parameter_domain = [[0.50000, 0.500001]]

    def u_ex(self, pre, xy, mu):
        x,y = xy
        if pre is dolfin:
            ln = pre.ln
        else:
            ln = pre.log
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
    
    def h_int(self, pre, xy, mu):
        assert self.version in [1,2,3]
        return 4.0/log(4.0) + 2.0
    
    def h_ext(self, pre, xy, mu): # dirichlet
        return 1.0
    
class TestCase6:
    def __init__(self,v=1):
        self.version = v 
        assert self.version == 1
        self.geometry = Donut1()
        self.nb_parameters = 1
        self.parameter_domain = [[0.50000, 0.500001]]

    def u_ex(self, pre, xy, mu):
        x,y = xy
        return pre.sin(x**2 + y**2)
    
    def grad_uex(self, pre, xy, mu):
        pass
    
    def grad2_uex(self, pre, xy, mu):
        pass
    
    def grad3_uex(self, pre, xy, mu):
        pass
    
    def f(self, pre, xy, mu):
        x,y = xy
        return (4.0 * (x**2 + y**2) + 1) * pre.sin(x**2 + y**2) - 4.0 * pre.cos(x**2 + y**2)
    
    def h_int(self, pre, xy, mu):
        return -cos(1.0/4.0)
    
    def h_ext(self, pre, xy, mu): # dirichlet
        return 2 * cos(1.0)
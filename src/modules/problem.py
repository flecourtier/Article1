from modules.geometry import Square1, UnitSquare, UnitCircle, Donut1

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
        self.geometry = UnitCircle() 
        if v>1:
            self.geometry = Donut1() 
        self.nb_parameters = 1
        self.parameter_domain = [[0.50000, 0.500001]]

    def u_ex(self, pre, xy, mu):
        pass

    def f(self, pre, xy, mu):
        return 1.0

    # Dirichlet BC        
    def g(self, pre, xy, mu):
        return 0.0
    
    # Neumann BC
    def h(self, pre, xy, mu):
        return 0.0
from modules.geometry import Square1

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
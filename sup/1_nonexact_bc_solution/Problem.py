import torch

class HomoSolOnUnitSquare:
    def __init__(self):
        self.parameter_domain = []

    def u_ex(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        return pre.sin(2*pre.pi*x)*pre.sin(2*pre.pi*y)
    
    def pert(self, pre, xy, mu):
        """Analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Analytical solution evaluated at (x,y)
        """
        x,y=xy
        return pre.cos(2*pre.pi*x)*pre.cos(2*pre.pi*y)
        # return pre.cos(x)

    def u_ex_prime(self, pre, xy, mu):
        """First derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: First derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        du_dx = 2*pre.pi*pre.cos(2*pre.pi*x)*pre.sin(2*pre.pi*y)
        du_dy = 2*pre.pi*pre.sin(2*pre.pi*x)*pre.cos(2*pre.pi*y)
        return du_dx,du_dy

    def u_ex_prime2(self, pre, xy, mu):
        """Second derivative of the analytical solution for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Second derivative of the analytical solution evaluated at (x,y)
        """
        x,y=xy
        du_dxx = -4*pre.pi**2*pre.sin(2*pre.pi*x)*pre.sin(2*pre.pi*y)
        du_dyy = -4*pre.pi**2*pre.sin(2*pre.pi*x)*pre.sin(2*pre.pi*y)
        return du_dxx,du_dyy

    def f(self, pre, xy, mu):
        """Right hand side of the PDE for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Right hand side of the PDE evaluated at (x,y)
        """
        x,y=xy
        return 8*pre.pi**2*pre.sin(2*pre.pi*x)*pre.sin(2*pre.pi*y)

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0*torch.ones_like(xy[0])
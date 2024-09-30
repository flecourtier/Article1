from testcases.geometry.geometry_3D import Cube1

class TestCase1_3D:
    def __init__(self):
        self.geometry = Cube1() 
        self.nb_parameters = 3
        self.parameter_domain = [[-0.5, 0.500001],[-0.50000, 0.500001],[-0.50000, 0.500001]]

    def u_ex(self, pre, xyz, mu):
        x,y,z=xyz
        mu1,mu2,mu3 = mu
        ex = pre.exp(-((x-mu1)**2.0 + (y-mu2)**2.0 + ((z-mu3)**2.0))/2)
        return ex * pre.sin(2*x) * pre.sin(2*y) * pre.sin(2*z)

    def f(self, pre, xyz, mu):
        x,y,z=xyz
        mu1,mu2,mu3 = mu
        ex = pre.exp(-((x-mu1)**2.0 + (y-mu2)**2.0 + ((z-mu3)**2.0))/2)
        return (4.0*(-mu1 + x)**1.0*pre.sin(2*y)*pre.sin(2*z)*pre.cos(2*x) - 1.0*(-mu1 + x)**2.0*pre.sin(2*x)*pre.sin(2*y)*pre.sin(2*z) + 4.0*(-mu2 + y)**1.0*pre.sin(2*x)*pre.sin(2*z)*pre.cos(2*y) - 1.0*(-mu2 + y)**2.0*pre.sin(2*x)*pre.sin(2*y)*pre.sin(2*z) + 4.0*(-mu3 + z)**1.0*pre.sin(2*x)*pre.sin(2*y)*pre.cos(2*z) - 1.0*(-mu3 + z)**2.0*pre.sin(2*x)*pre.sin(2*y)*pre.sin(2*z) + 15.0*pre.sin(2*x)*pre.sin(2*y)*pre.sin(2*z))*ex

    def g(self, pre, xy, mu):
        """Boundary condition for the Circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :param mu: (S) parameter
        :return: Boundary condition evaluated at (x,y)
        """
        return 0.0
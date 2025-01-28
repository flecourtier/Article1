# Test Donut - Conditions Mixtes présentés dans le papier (paramétrique) + Loss Sobolev
# Conditions exactes
# FONCTIONNE

from pathlib import Path

import matplotlib.pyplot as plt
import scimba.equations.domain as domain
import scimba.nets.training_tools as training_tools
import scimba.pinns.pinn_losses as pinn_losses
import scimba.pinns.pinn_x as pinn_x
import scimba.pinns.training_x as training_x
import scimba.sampling.sampling_parameters as sampling_parameters
import scimba.sampling.sampling_pde as sampling_pde
import scimba.sampling.uniform_sampling as uniform_sampling
import torch
from scimba.equations import pdes

from testcases.geometry.geometry_2D import Donut
from testcases.problem.problem_2D import TestCase5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846

current = Path(__file__).parent.parent.parent.parent.parent.parent
current_filename = Path(__file__).name

current_testcase = int(current_filename.split("test_")[1].split("_")[0])
current_version = int(current_filename.split("_v")[1].split(".")[0])

def create_fulldomain(geometry):
    bigcenter = geometry.bigcircle.center
    bigradius = geometry.bigcircle.radius
    smallcenter = geometry.hole.center
    smallradius = geometry.hole.radius
    
    print("bigcenter : ",bigcenter)
    print("bigradius : ",bigradius)
    print("smallcenter : ",smallcenter)
    print("smallradius : ",smallradius)
    
    class BigDomain(domain.SignedDistance):
        def __init__(self):
            super().__init__(dim=2)

        def sdf(self, x):
            x1, x2 = x.get_coordinates()
            return torch.sqrt((x1 - bigcenter[0]) ** 2 + (x2 - bigcenter[1]) ** 2) - bigradius
    
    class Hole(domain.SignedDistance):
        def __init__(self):
            super().__init__(dim=2)

        def sdf(self, x):
            x1, x2 = x.get_coordinates()
            return torch.sqrt((x1 - smallcenter[0]) ** 2 + (x2 - smallcenter[1]) ** 2) - smallradius
        
    sdf = BigDomain()
    sdf_hole = Hole()
    xdomain = domain.SignedDistanceBasedDomain(2, [[-1.0, 1.0], [-1.0, 1.0]], sdf)
    hole = domain.SignedDistanceBasedDomain(2, [[-1.0, 1.0], [-1.0, 1.0]], sdf_hole)
    
    fulldomain = domain.SpaceDomain(2, xdomain)
    fulldomain.add_hole(hole)
    
    # to plot bc
    def big(t):
        return torch.cat([
            bigcenter[0] + bigradius*torch.cos(2.0 * PI * t), 
            bigcenter[0] + bigradius*torch.sin(2.0 * PI * t)], 
        axis=1)

    def small(t):
        return torch.cat([
            smallcenter[0] + smallradius*torch.cos(2.0 * PI * t), 
            smallcenter[0] + smallradius*torch.sin(2.0 * PI * t)], 
        axis=1)
    
    bc_big = domain.ParametricCurveBasedDomain(2, [[0.0, 1.0]], big)
    fulldomain.add_bc_subdomain(bc_big)
    bc_hole = domain.ParametricCurveBasedDomain(2, [[0.0, 1.0]], small)
    hole.add_bc_subdomain(bc_hole)
    
    
    return fulldomain,xdomain,hole

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase5(version=current_version)     
        assert isinstance(self.problem.geometry, Donut)
        
        space_domain,_,_ = create_fulldomain(self.problem.geometry)
        
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=self.problem.parameter_domain,
        )

        self.first_derivative = True
        self.second_derivative = True
        self.third_derivative = True
        # self.compute_normals = True
        self.coeff_third_derivative = 0.1

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs): 
        pass       
    
    # def residual(self, w, x, mu, **kwargs):
    #     x1, x2 = x.get_coordinates()
    #     u_xx = self.get_variables(w, "w_xx")
    #     u_yy = self.get_variables(w, "w_yy")
    #     f = self.problem.f(torch, [x1, x2], mu)
        
    #     return u_xx + u_yy + f
    
    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        mu = self.get_parameters(mu)

        # compute residual
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, [x1, x2], [mu])
        
        res = u_xx + u_yy + f

        # compute d/dx and d/dy residual
        df_dx, df_dy = self.problem.gradf(torch, [x1, x2], [mu])
        
        u_xxx = self.get_variables(w, "w_xxx")
        u_xyy = self.get_variables(w, "w_xyy")

        dres_dx = u_xxx + u_xyy + df_dx

        u_xxy = self.get_variables(w, "w_xxy")
        u_yyy = self.get_variables(w, "w_yyy")

        dres_dy = u_xxy + u_yyy + df_dy

        return torch.sqrt(
            res**2 + self.coeff_third_derivative * (dres_dx**2 + dres_dy**2)
        )
    
    def post_processing(self, x, mu, w):   
        x1,x2 = x.get_coordinates()

        # compute levelset
        phi_E = -self.space_domain.large_domain.sdf(x)
        phi_I = self.space_domain.list_holes[0].sdf(x)
        phi = (phi_E * phi_I) / (phi_E + phi_I)
        
        # get BC condition
        h = self.problem.h_int(torch, [x1,x2], mu)
        g = self.problem.h_ext(torch, [x1,x2], mu)
        
        ones = torch.ones_like(x1)
        gradphi_I = torch.autograd.grad(phi_I, x.x, ones, create_graph=True)[0]
        gradw = torch.autograd.grad(w, x.x, ones, create_graph=True)[0] #, allow_unused=True)
        
        w1 = phi_E / (phi_E + phi_I**2)
        w2 = phi_I**2 / (phi_E + phi_I**2)
        
        u1 = g
        element_wise_product = gradphi_I * gradw
        dot_product = torch.sum(element_wise_product, dim=1)[:,None]
        u2 = w + phi_I * (w - dot_product) - phi_I * h
        
        # Somme des produits le long de la dimension 1
        res = w1 * u2 + w2 * u1 + phi_E * phi_I**2 * w
        res = res.reshape(-1,1)

        return res

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        return self.problem.u_ex(torch, [x1,x2], mu)
    
    def reference_solution_derivative(self, x, mu):
        x1,x2 = x.get_coordinates()
        return self.problem.gradu_ex(torch, [x1,x2], mu)

    def reference_solution_second_derivative(self, x, mu):
        x1,x2 = x.get_coordinates()
        return self.problem.grad2u_ex(torch, [x1,x2], mu)

def Run_laplacian2D(pde,new_training=False,plot_bc=False):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = current / "networks" / "test_2D" / f"test_fe{current_testcase}_v{current_version}.pth"

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    if plot_bc:
        x, mu = sampler.bc_sampling(1000)
        x1, x2 = x.get_coordinates(label=0)
        plt.scatter(x1.cpu().detach().numpy(), x2.cpu().detach().numpy(), color="b", label="Dir")
        x1, x2 = x.get_coordinates(label=1)
        plt.scatter(x1.cpu().detach().numpy(), x2.cpu().detach().numpy(), color="r", label="Rob")
        plt.legend()
        plt.show()

    tlayers = [40, 60, 80, 60, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)

    losses = pinn_losses.PinnLossesData(bc_loss_bool=False, w_res=1.0, w_bc=0.0)
    optimizers = training_tools.OptimizerData(learning_rate=8e-3, decay=0.99, switch_to_LBFGS=True, switch_to_LBFGS_at=3000,LBFGS_switch_plateau=[3000,10])

    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=8000,
    )

    if new_training:
        trainer.train(epochs=4000, n_collocation=6000, n_bc_collocation=8000)
        # trainer.train(epochs=1, n_collocation=8000, n_bc_collocation=8000)

    filename = current / "networks" / "test_2D" / f"test_fe{current_testcase}_v{current_version}.png"
    trainer.plot(20000,filename=filename,reference_solution=True,random=True)
    
    return trainer,pinn

def check_BC():
    geometry = pde.problem.geometry

    bigcenter = geometry.bigcircle.center
    bigradius = geometry.bigcircle.radius
    smallcenter = geometry.hole.center
    smallradius = geometry.hole.radius
    
    import numpy as np
    from scimba.equations.domain import SpaceTensor
    
    def big(t):
        return [bigcenter[0] + bigradius*np.cos(2.0 * PI * t), 
        bigcenter[0] + bigradius*np.sin(2.0 * PI * t)]

    def small(t):
        return [smallcenter[0] + smallradius*np.cos(2.0 * PI * t), 
        smallcenter[0] + smallradius*np.sin(2.0 * PI * t)]

    t = np.linspace(0,1,10)

    XY_big = np.array(big(t)).T
    XY_small = np.array(small(t)).T
    
    # check Neumann on big circle
    def check(which="big"):
        assert which in ["big","small"]
        
        if which == "big":
            XY = XY_big
        else:
            XY = XY_small
            
        # get points on the boundary, parameters and evaluate u_theta
        X_test = torch.tensor(XY,requires_grad=True)
        X_test = SpaceTensor(X_test,torch.zeros_like(X_test,dtype=int))
        
        nb_params = len(trainer.pde.parameter_domain)
        shape = (XY.shape[0],nb_params)
        ones = torch.ones(shape)
        mu = (pde.problem.parameter_domain[0][0]+pde.problem.parameter_domain[0][1])/2
        print(mu)
        mu_test = torch.Tensor([mu]).to(device) * ones.to(device)
        # (torch.Tensor([0.5]).to(device) * ones).to(device)
        
        u_theta = pinn.setup_w_dict(X_test, mu_test)["w"][:,0].reshape(-1,1)
        
        # compute Dirichlet condition
        if which == "big":
            phi = pde.space_domain.large_domain.sdf(X_test)
            print("Dirichlet : ",u_theta.reshape(-1))
        # compute Robin condition
        else:
            grad_u_theta = torch.autograd.grad(u_theta, X_test.x, ones, create_graph=True)[0]
            phi = -pde.space_domain.list_holes[0].sdf(X_test)
            gradphi = torch.autograd.grad(phi, X_test.x, ones, create_graph=True)[0]
            
            element_wise_product = gradphi * grad_u_theta
            dot_product = torch.sum(element_wise_product, dim=1)[:,None]
            bc_Robin = dot_product + u_theta
            print("Robin : ",bc_Robin.reshape(-1))
        
        from math import log
        if which == "big":
            dir = pde.problem.g(torch, XY, [mu])
            print("ex Dirichlet : ",dir)
        else:
            neu = pde.problem.gr(torch, XY, [mu])
            print("ex Robin : ",neu)
        
    print("## Values for Neumann condition on big circle")
    check("big")
    print("## Values for Neumann condition on small circle")
    check("small")

if __name__ == "__main__":
    pde = Poisson_2D()
    trainer, pinn = Run_laplacian2D(pde,new_training=True,plot_bc=False)

    check_BC()
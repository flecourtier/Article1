# Test Donut - Conditions Neumann partout (non paramétrique)
# Conditions exactes sur les deux cercles (levelsel, pas de loss BC)
# NE FONCTIONNE PAS

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

from modules.geometry import Donut
from modules.problem import TestCase5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846

current = Path(__file__).parent.parent.parent.parent.parent.parent

def create_fulldomain(geometry):
    bigcenter = geometry.bigcircle.center
    bigradius = geometry.bigcircle.radius
    smallcenter = geometry.hole.center
    smallradius = geometry.hole.radius
    
    class BigDomain(domain.SignedDistance):
        def __init__(self):
            super().__init__(dim=2)

        def sdf(self, x):
            x1, x2 = x.get_coordinates()
            return (x1 - bigcenter[0]) ** 2 + (x2 - bigcenter[1]) ** 2 - bigradius**2
    
    class Hole(domain.SignedDistance):
        def __init__(self):
            super().__init__(dim=2)

        def sdf(self, x):
            x1, x2 = x.get_coordinates()
            return (x1 - smallcenter[0]) ** 2 + (x2 - smallcenter[1]) ** 2 - smallradius**2
        
    sdf = BigDomain()
    sdf_hole = Hole()
    xdomain = domain.SignedDistanceBasedDomain(2, [[-1.0, 1.0], [-1.0, 1.0]], sdf)
    hole = domain.SignedDistanceBasedDomain(2, [[-1.0, 1.0], [-1.0, 1.0]], sdf_hole)
    
    fulldomain = domain.SpaceDomain(2, xdomain)
    fulldomain.add_hole(hole)
    
    return fulldomain,xdomain,hole

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase5(v=2)
        
        assert isinstance(self.problem.geometry, Donut)
        
        space_domain,_,_ = create_fulldomain(self.problem.geometry)
        
        super().__init__(
            nb_unknowns=2,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=self.problem.parameter_domain,
        )

        self.first_derivative = True
        self.second_derivative = True
        # self.compute_normals = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs): 
        pass       
    
    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        u,_ = self.get_variables(w)
        u_xx,_ = self.get_variables(w, "w_xx")
        u_yy,_ = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, [x1, x2], mu)
        
        return u_xx + u_yy - u + f
    
    def post_processing(self, x, mu, w):   
        x1,x2 = x.get_coordinates()
             
        # compute levelsets
        phi_E = self.space_domain.large_domain.sdf(x)
        phi_I = self.space_domain.list_holes[0].sdf(x)
        phi = phi_I * phi_E
        ones = torch.ones_like(x1)
        gradphi = torch.autograd.grad(phi, x.x, ones, create_graph=True)[0]
        
        # get u1 and u2
        u1 = w[:,0].reshape(-1,1)
        u2 = w[:,1].reshape(-1,1)
        
        ones = torch.ones_like(u1)
        gradu1 = torch.autograd.grad(u1, x.x, ones, create_graph=True)[0] #, allow_unused=True)
        
        # get h
        h_I = self.problem.h_int(torch, [x1,x2], mu)
        h_E = self.problem.h_ext(torch, [x1,x2], mu)
        sumphi = phi_I**2+phi_E**2
        h_glob = phi_I**2 / sumphi * h_E + phi_E**2 / sumphi * h_I
        
        # Produit élément par élément
        element_wise_product = gradphi * gradu1

        # Somme des produits le long de la dimension 1
        dot_product = torch.sum(element_wise_product, dim=1)[:,None]     
        res = (u1 + phi*dot_product) - phi * h_glob + phi**2 * u2
        res = res.reshape(-1,1)

        res2 = torch.cat([res,res],dim=1)
        return res2

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        return self.problem.u_ex(torch, [x1,x2], mu)

def Run_laplacian2D(pde,new_training=False,plot_bc=False):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = current / "networks" / "test_fe5_v2.pth"

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    if plot_bc:
        x, mu = sampler.bc_sampling(1000)
        x1, x2 = x.get_coordinates(label=0)
        plt.scatter(x1.cpu().detach().numpy(), x2.cpu().detach().numpy(), color="r", label="Neu")
        x1, x2 = x.get_coordinates(label=1)
        plt.scatter(x1.cpu().detach().numpy(), x2.cpu().detach().numpy(), color="b", label="Neu")
        plt.legend()
        plt.show()

    tlayers = [40, 40, 40, 40, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)

    losses = pinn_losses.PinnLossesData(bc_loss_bool=False, w_res=1.0, w_bc=0.0)
    optimizers = training_tools.OptimizerData(learning_rate=1.0e-2, decay=0.99)

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
        trainer.train(epochs=1000, n_collocation=8000, n_bc_collocation=8000)
        # trainer.train(epochs=1, n_collocation=8000, n_bc_collocation=8000)

    filename = current / "networks" / "test_fe5_v2.png"
    trainer.plot(20000,filename=filename,reference_solution=True)
    
    return trainer,pinn

if __name__ == "__main__":
    pde = Poisson_2D()
    trainer, pinn = Run_laplacian2D(pde,new_training=False,plot_bc=False)

    geometry = pde.problem.geometry

    bigcenter = geometry.bigcircle.center
    bigradius = geometry.bigcircle.radius
    smallcenter = geometry.hole.center
    smallradius = geometry.hole.radius
    
    import numpy as np
    def big(t):
        return [bigcenter[0] + bigradius*np.cos(2.0 * PI * t), 
        bigcenter[0] + bigradius*np.sin(2.0 * PI * t)]

    def small(t):
        return [smallcenter[0] + smallradius*np.cos(2.0 * PI * t), 
        smallcenter[0] + smallradius*np.sin(2.0 * PI * t)]

    t = np.linspace(0,1,100)

    XY_big = np.array(big(t)).T
    
    from scimba.equations.domain import SpaceTensor
    X_test = torch.tensor(XY_big,requires_grad=True)
    X_test = SpaceTensor(X_test,torch.zeros_like(X_test,dtype=int))

    # get parameters
    nb_params = len(trainer.pde.parameter_domain)
    shape = (XY_big.shape[0],nb_params)
    ones = torch.ones(shape)
    mu_test = (torch.Tensor([0.5]).to(device) * ones).to(device)
    
    u_theta = pinn.setup_w_dict(X_test, mu_test)["w"][:,0].reshape(-1,1)
    print(u_theta.shape)
    ones = torch.ones_like(u_theta)
    grad_u_theta = torch.autograd.grad(u_theta, X_test.x, ones, create_graph=True)[0].cpu()

    # check BC Neumann
    normals = torch.Tensor(XY_big.copy())
    element_wise_product = grad_u_theta * normals
    dot_product = torch.sum(element_wise_product, dim=1)[:,None]
    print("<grad_u,n> :", dot_product)
    exact = 2*np.cos(1.0)*torch.ones_like(dot_product)
    diff = dot_product - exact
    print("diff.std() :",diff.std())
    print("diff.mean() :",diff.mean())
# Test Donut - Conditions mixtes (non paramétrique)
# Conditions non exact sur le bord (loss BC, pas de level set à  cause de Neumann)

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
from modules.problem import TestCase4

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
    # domain creation
    xdomain = domain.DiskBasedDomain(2, bigcenter, bigradius)
    # hole = domain.DiskBasedDomain(2, geometry.hole.center, geometry.hole.radius)
    
    class Hole(domain.SignedDistance):
        def __init__(self):
            super().__init__(dim=2)

        def sdf(self, x):
            x1, x2 = x.get_coordinates()
            return (x1 - smallcenter[0]) ** 2 + (x2 - smallcenter[1]) ** 2 - smallradius**2
        
    sdf = Hole()
    hole = domain.SignedDistanceBasedDomain(2, [[-1.0, 1.0], [-1.0, 1.0]], sdf)
    
    fulldomain = domain.SpaceDomain(2, xdomain)
    fulldomain.add_hole(hole)
    
    def big(t):
        center = geometry.bigcircle.center
        radius = geometry.bigcircle.radius
        return torch.cat([
            center[0] + radius*torch.cos(2.0 * PI * t), 
            center[0] + radius*torch.sin(2.0 * PI * t)], 
        axis=1)

    def small(t):
        center = geometry.hole.center
        radius = geometry.hole.radius
        return torch.cat([
            center[0] + radius*torch.cos(2.0 * PI * t), 
            center[0] + radius*torch.sin(2.0 * PI * t)], 
        axis=1)

    bc_Dir = domain.ParametricCurveBasedDomain(2, [[0.0, 1.0]], small)
    bc_Neu = domain.ParametricCurveBasedDomain(2, [[0.0, 1.0]], big)

    fulldomain.add_bc_subdomain(bc_Dir)
    fulldomain.add_bc_subdomain(bc_Neu)
    
    return fulldomain

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase4(v=2)
        
        assert isinstance(self.problem.geometry, Donut)
        
        space_domain = create_fulldomain(self.problem.geometry)
        
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=self.problem.parameter_domain,
        )

        self.first_derivative = True
        self.second_derivative = True
        self.compute_normals = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):        
        u_top = self.get_variables(w, label=0)
        x1, x2 = x.get_coordinates(label=0)
        g = self.problem.g(torch, [x1,x2], mu)
        
        u_x_bottom = self.get_variables(w, "w_x", label=1)
        u_y_bottom = self.get_variables(w, "w_y", label=1)
        n_x, n_y = x.get_normals(label=1)
        x1, x2 = x.get_coordinates(label=1)
        h = self.problem.h(torch, [x1,x2], mu)
        
        return u_top - g , u_x_bottom * n_x + u_y_bottom * n_y - h

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, [x1, x2], mu)
        return u_xx + u_yy + f

    # def reference_solution(self, x, mu):
    #     x1, x2 = x.get_coordinates()
    #     return 0.0 * x1

def Run_laplacian2D(pde,training=False,plot_bc=False):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = current / "networks" / "test_fe4_v2.pth"
    new_training = False
    # new_training = True

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    if plot_bc:
        x, mu = sampler.bc_sampling(1000)
        x1, x2 = x.get_coordinates(label=0)
        plt.scatter(x1.cpu().detach().numpy(), x2.cpu().detach().numpy(), color="r", label="Dir")
        x1, x2 = x.get_coordinates(label=1)
        plt.scatter(x1.cpu().detach().numpy(), x2.cpu().detach().numpy(), color="b", label="Neu")
        plt.legend()
        plt.show()

    tlayers = [40, 40, 40, 40, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)

    losses = pinn_losses.PinnLossesData(bc_loss_bool=True, w_res=1.0, w_bc=30.0)
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

    if training:
        trainer.train(epochs=750, n_collocation=8000, n_bc_collocation=8000)

    filename = current / "networks" / "test_fe4_v2.png"
    trainer.plot(20000,filename=filename)
    
    return trainer,pinn

if __name__ == "__main__":
    pde = Poisson_2D()
    network, trainer = Run_laplacian2D(pde,training=False,plot_bc=True)
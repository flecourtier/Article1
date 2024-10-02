import time
from pathlib import Path
import torch
from torch.autograd import grad
import scimba.nets.training_tools as training_tools
import scimba.pinns.pinn_losses as pinn_losses
import scimba.pinns.pinn_x as pinn_x
import scimba.pinns.training_x as training_x
import scimba.sampling.sampling_parameters as sampling_parameters
import scimba.sampling.sampling_pde as sampling_pde
import scimba.sampling.uniform_sampling as uniform_sampling
from scimba.equations import domain, pdes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")
torch.set_default_dtype(torch.double)
torch.set_default_device(device)

current = Path(__file__).parent.parent.parent.parent.parent.parent

PI = 3.14159265358979323846
# test from 10.3390/en15186823

from testcases.problem.problem_2D import TestCase3_new

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase3_new()
        space_domain = domain.SpaceDomain(2, domain.SquareDomain(2, self.problem.geometry.box))
        
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=self.problem.parameter_domain,
        )
        self.first_derivative = True
        self.second_derivative = True
        def anisotropy_matrix(w, x, mu):
            x1, x2 = x.get_coordinates()
            c1, c2, sigma, eps = self.get_parameters(mu)
            a11 = eps * x1**2 + x2**2
            a12 = (eps - 1) * x1 * x2
            a21 = (eps - 1) * x1 * x2
            a22 = x1**2 + eps * x2**2
            return torch.cat((a11, a12, a21, a22), axis=1)
        self.anisotropy_matrix = anisotropy_matrix
        self.coeff_dx3 = 1e-3
        self.coeff_dmu = 0.5e-3
        
    def make_data(self, n_data):
        pass
    
    def bc_residual(self, w, x, mu, **kwargs):
        return self.get_variables(w)
    
    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        c1, c2, sigma, eps = self.get_parameters(mu)
        div_K_grad_u = self.get_variables(w, "div_K_grad_w")
        f = 10 * torch.exp(-((x1 - c1) ** 2 + (x2 - c2) ** 2) / (0.025 * sigma**2))
        res = div_K_grad_u + f
        ones = torch.ones_like(f)
        d_div_K_grad_u_dx = grad(div_K_grad_u, x.x, ones, retain_graph=True)[0]
        d_f_dx = grad(f, x.x, ones, retain_graph=True)[0]
        d_res_dx = torch.sum(d_div_K_grad_u_dx + d_f_dx, dim=1)[:, None]
        d_div_K_grad_u_dmu = grad(div_K_grad_u, mu, ones, retain_graph=True)[0]
        d_f_dmu = grad(f, mu, ones, retain_graph=True)[0]
        d_res_dmu = torch.sum(d_div_K_grad_u_dmu + d_f_dmu, dim=1)[:, None]
        return torch.sqrt(
            res**2 + self.coeff_dx3 * d_res_dx**2 + self.coeff_dmu * d_res_dmu**2
        )
        
    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        return w * 0.1 * torch.sin(PI * x1) * torch.sin(PI * x2)
    
def Run_laplacian2D(pde,new_training = False):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)
    file_name = current / "networks" / "test_2D" / "test_fe3_new.pth"

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)
    tlayers = [80, 80, 160, 80, 80]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    # network = pinn_x.RBFNet_x(pde=pde,
    #                        sampler=sampler,
    #                       nb_func=30,
    #                        type_g="anistropic")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=False, w_res=1.0, w_bc=500.0, adaptive_weights="annealing"
    )
    optimizers = training_tools.OptimizerData(
        learning_rate=7e-3,
        # learning_rate=5e-3,
        # learning_rate=1e-3,
        # learning_rate=1.5e-2,
        decay=0.99,
        # switch_to_LBFGS=True,
        # switch_to_LBFGS_at=1,
    )
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=100000,
    )
    if new_training:
        start = time.perf_counter()
        trainer.train(epochs=5_000, n_collocation=15_000, n_bc_collocation=0, n_data=0)
        print("training time:", time.perf_counter() - start)
        
    return trainer, pinn

if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    pde = Poisson_2D()
    trainer, network = Run_laplacian2D(pde)
    trainer.plot(n_visu=20000)
    # plot for 3 random sets of parameters
    for _ in range(3):
        trainer.plot_2d_contourf(draw_contours=True, random=True)
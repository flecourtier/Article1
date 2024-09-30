from pathlib import Path

import scimba.nets.training_tools as training_tools
import scimba.pinns.pinn_losses as pinn_losses
import scimba.pinns.pinn_x as pinn_x
import scimba.pinns.training_x as training_x
import scimba.sampling.sampling_parameters as sampling_parameters
import scimba.sampling.sampling_pde as sampling_pde
import scimba.sampling.uniform_sampling as uniform_sampling
import torch
from scimba.equations import domain, pdes

from testcases.geometry.geometry_2D import Square
from testcases.problem.problem_2D import TestCase3, TestCase3_small_param, TestCase3_medium_param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846

# test from 10.3390/en15186823
 
current = Path(__file__).parent.parent.parent.parent.parent.parent


class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self,size_param="big"):
        assert size_param in ["small", "medium", "big"]
        
        if size_param == "big":
            self.problem = TestCase3()
        elif size_param == "medium":
            self.problem = TestCase3_medium_param()
        else:
            self.problem = TestCase3_small_param()
        
        assert isinstance(self.problem.geometry, Square)
        
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

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        return self.get_variables(w)

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        c1, c2, sigma, eps = self.get_parameters(mu)
        div_K_grad_u = self.get_variables(w, "div_K_grad_w")
        f = torch.exp(-((x1 - c1) ** 2 + (x2 - c2) ** 2) / (0.025 * sigma**2))
        return div_K_grad_u + f

    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        return (1 - x1) * x1 * x2 * (1 - x2) * w


def Run_laplacian2D(pde, size_param="big", new_training = False, largenet=False):
    assert size_param in ["small", "medium", "big"]
    
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    namefe3 = "test_fe3"
    if size_param != "big":
        namefe3 += "_"+size_param+"_param"
    if largenet and size_param == "medium":
        namefe3 += "_largenet"
        
    file_name = current / "networks" / (namefe3+".pth")

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    if size_param == "big":
        tlayers = [40, 60, 60, 60, 40]
    else:
        if largenet and size_param == "medium":
            tlayers = [120]*5
        else:
            tlayers = [80]*5
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData()
    optimizers = training_tools.OptimizerData(
        learning_rate=1.6e-2,
        decay=0.99,
        # switch_to_LBFGS=True,
        # switch_to_LBFGS_at=1000,
    )
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=15000,
    )

    if new_training:
        trainer.train(epochs=1000, n_collocation=8000, n_bc_collocation=0, n_data=0)

    filename = current / "networks" / (namefe3+".png")
    trainer.plot(50000, random=True, filename=filename)
    
    return trainer, pinn


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0, 1.0], [0.0, 1.0]]))
    print(xdomain)
    pde = Poisson_2D(xdomain)

    network, trainer = Run_laplacian2D(pde,new_training = False)

    # test contour plots on square
    trainer.plot_2d_contourf(draw_contours=True)
    trainer.plot_2d_contourf(draw_contours=True, residual=True, n_visu=256)

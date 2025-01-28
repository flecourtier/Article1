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
from testcases.problem.problem_2D import TestCase3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

current = Path(__file__).parent.parent.parent.parent.parent.parent
current_filename = Path(__file__).name

current_testcase = int(current_filename.split("test_")[1].split("_")[0])
current_version = int(current_filename.split("_v")[1].split(".")[0])

class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase3(version=current_version)
        
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

            return torch.cat(self.problem.anisotropy_matrix(torch, [x1, x2], [c1, c2, sigma, eps]), axis=1)

        self.anisotropy_matrix = anisotropy_matrix

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        return self.get_variables(w)

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        c1, c2, sigma, eps = self.get_parameters(mu)
        div_K_grad_u = self.get_variables(w, "div_K_grad_w")
        f = self.problem.f(torch, [x1, x2], [c1, c2, sigma, eps])
        return div_K_grad_u + f

    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        return (1 - x1) * x1 * x2 * (1 - x2) * w


def Run_laplacian2D(pde, new_training = False, largenet=False):
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

    tlayers = [40, 60, 60, 60, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData()
    optimizers = training_tools.OptimizerData(
        learning_rate=1.6e-2,
        decay=0.99,
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
        trainer.train(epochs=15000, n_collocation=8000, n_bc_collocation=0, n_data=0)

    filename = current / "networks" / "test_2D" / f"test_fe{current_testcase}_v{current_version}.png"
    trainer.plot(50000, random=True, filename=filename)
    
    return trainer, pinn


if __name__ == "__main__":
    pde = Poisson_2D()
    Run_laplacian2D(pde,new_training = False)
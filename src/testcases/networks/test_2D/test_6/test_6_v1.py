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

from testcases.geometry.geometry_2D import Circle
from testcases.problem.problem_2D import TestCase6

current = Path(__file__).parent.parent.parent.parent.parent.parent
current_filename = Path(__file__).name

current_testcase = int(current_filename.split("test_")[1].split("_")[0])
current_version = int(current_filename.split("_v")[1].split(".")[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

class NonlinearPoisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase6(version=current_version)
        
        assert isinstance(self.problem.geometry, Circle)
        
        center = self.problem.geometry.center
        radius = self.problem.geometry.radius
        space_domain = domain.SpaceDomain(2, domain.DiskBasedDomain(2, [center[0], center[1]], radius))
        
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=self.problem.parameter_domain,
        )

        self.first_derivative = True
        self.second_derivative = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        return self.get_variables(w)
    
    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        mu1 = self.get_parameters(mu)

        u = self.get_variables(w, "w")
        u_x = self.get_variables(w, "w_x")
        u_y = self.get_variables(w, "w_y")
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")

        # print(mu1)
        # x1_0, x2_0 = self.space_domain.large_domain.center
        # phi = (x1 - x1_0) ** 2 + (x2 - x2_0) ** 2 -1
        # rhs = (2 * mu1**2 - 0.5 * mu1) * phi + mu1**2
        
        f = self.problem.f(torch, [x1, x2], [mu1])

        return 2 * u * (u_x**2+u_y**2) + (1+u**2)*(u_xx + u_yy) + f


    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        center = self.problem.geometry.center
        radius = self.problem.geometry.radius
        
        phi = (x1 - center[0]) ** 2 + (x2 - center[1]) ** 2 - radius ** 2
        
        return phi*w

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        x1_0, x2_0 = self.space_domain.large_domain.center
        mu1 = self.get_parameters(mu)
        # return 0.5 * mu1 * (1 - (x1 - x1_0) ** 2 - (x2 - x2_0) ** 2)
        return self.problem.u_ex(torch, [x1, x2], [mu1])


def Run_laplacian2D(pde, new_training = False):
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

    tlayers = [20, 40, 40, 20]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=False, w_res=1.0, w_bc=10.0
    )
    optimizers = training_tools.OptimizerData(learning_rate=5e-2, decay=0.99)
    
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=2000,
    )

    if new_training:
        trainer.train(epochs=1000, n_collocation=2000, n_bc_collocation=0)

    filename = current / "networks" / "test_2D" / f"test_fe{current_testcase}_v{current_version}.png"
    trainer.plot(20000, random=True,reference_solution=True, filename=filename)
    # trainer.plot_derivative_mu(n_visu=20000)
    
    return trainer,pinn


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    
    pde = NonlinearPoisson_2D()

    Run_laplacian2D(pde,new_training = True)
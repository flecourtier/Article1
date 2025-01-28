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

from testcases.geometry.geometry_1D import Line
from testcases.problem.problem_1D import TestCase2

current = Path(__file__).parent.parent.parent.parent.parent.parent
current_filename = Path(__file__).name

current_testcase = int(current_filename.split("test_")[1].split("_")[0])
current_version = int(current_filename.split("_v")[1].split(".")[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

class Poisson_1D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase2()
        
        assert isinstance(self.problem.geometry, Line)
        
        space_domain = domain.SpaceDomain(1, domain.SquareDomain(1, self.problem.geometry.box))
        
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
        pass
    
    def residual(self, w, x, mu, **kwargs):
        r, Pe = self.get_parameters(mu)
        u_x = self.get_variables(w, "w_x")
        u_xx = self.get_variables(w, "w_xx")
        
        return u_x - 1.0/Pe * u_xx - r

    def post_processing(self, x, mu, w):
        x1 = x.get_coordinates()
        
        a = self.problem.geometry.box[0][0]
        b = self.problem.geometry.box[0][1]
        
        return (x1-a)*(x1-b)*w
    
    def reference_solution(self, x, mu):
        x1 = x.get_coordinates()
        r, Pe = self.get_parameters(mu)
        return self.problem.u_ex(torch, x1, [r,Pe])
    
    # def reference_solution_derivative(self, x, mu):
    #     x1 = x.get_coordinates()
    #     alpha, beta, gamma = self.get_parameters(mu)
    #     return self.problem.du_ex_dx(torch, x1, [alpha, beta, gamma])

    # def reference_solution_second_derivative(self, x, mu):
    #     x1 = x.get_coordinates()
    #     alpha, beta, gamma = self.get_parameters(mu)
    #     return self.problem.d2u_ex_dx2(torch, x1, [alpha, beta, gamma])


def Run_laplacian1D(pde, bc_loss_bool=False, new_training = False):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = current / "networks" / "test_1D" / f"test_fe{current_testcase}_v{current_version}.pth"

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    tlayers = [40, 40, 40, 40, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=bc_loss_bool, w_res=1.0, w_bc=0.0
    )
    optimizers = training_tools.OptimizerData(learning_rate=1.0e-3, decay=0.99)
    
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=5000,
    )

    if new_training:
        trainer.train(
            epochs=30000, n_collocation=5000
        )

    filename = current / "networks" / "test_1D" / f"test_fe{current_testcase}_v{current_version}.png"
    trainer.plot(20000, random=True,reference_solution=True, filename=filename)
    # trainer.plot_derivative_mu(n_visu=20000)
    
    return trainer,pinn


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    pde = Poisson_1D()
    Run_laplacian1D(pde,new_training = False)
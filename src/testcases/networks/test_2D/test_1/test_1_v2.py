# meilleurs apprentissages des dérivées troisièmes

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
from testcases.problem.problem_2D import TestCase1

import matplotlib.pyplot as plt

current = Path(__file__).parent.parent.parent.parent.parent.parent
current_filename = Path(__file__).name

current_testcase = int(current_filename.split("test_")[1].split("_")[0])
current_version = int(current_filename.split("_v")[1].split(".")[0])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)


class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase1(version=current_version)
        assert self.problem.testcase == current_testcase

        assert isinstance(self.problem.geometry, Square)

        space_domain = domain.SpaceDomain(
            2, domain.SquareDomain(2, self.problem.geometry.box)
        )

        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=self.problem.parameter_domain,
        )

        self.first_derivative = True
        self.second_derivative = True
        self.third_derivative = True

        self.coeff_third_derivative = 0.1

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        u = self.get_variables(w)
        return u

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        mu1, mu2 = self.get_parameters(mu)

        # compute residual
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = self.problem.f(torch, [x1, x2], [mu1, mu2])

        res = u_xx + u_yy + f

        # compute d/dx and d/dy residual
        df_dx, df_dy = self.problem.gradf(torch, [x1, x2], [mu1, mu2])

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
        x1, x2 = x.get_coordinates()

        a = self.problem.geometry.box[0][0]
        b = self.problem.geometry.box[0][1]
        a2 = self.problem.geometry.box[1][0]
        b2 = self.problem.geometry.box[1][1]

        return (x1 - a) * (b - x1) * (x2 - a2) * (b2 - x2) * w

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        mu1, mu2 = self.get_parameters(mu)
        return self.problem.u_ex(torch, [x1, x2], [mu1, mu2])


def Run_laplacian2D(pde, new_training=False):
    print(current)
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = (
        current
        / "networks"
        / "test_2D"
        / f"test_fe{current_testcase}_v{current_version}.pth"
    )

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    tlayers = [40, 60, 60, 60, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(bc_loss_bool=False, w_res=1.0, w_bc=0.0)
    optimizers = training_tools.OptimizerData(learning_rate=1.7e-2, decay=0.99)

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
        trainer.train(epochs=3000, n_collocation=6000, n_data=0)

    filename = (
        current
        / "networks"
        / "test_2D"
        / f"test_fe{current_testcase}_v{current_version}.png"
    )
    trainer.plot(20000, random=True, reference_solution=True, filename=filename)
    # trainer.plot_derivative_mu(n_visu=20000)

    return trainer, pinn


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn

    pde = Poisson_2D()
    Run_laplacian2D(pde, new_training=False)
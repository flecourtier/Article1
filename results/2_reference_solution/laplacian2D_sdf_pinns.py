from pathlib import Path

import scimba.nets.training_tools as training_tools
import scimba.pinns.pinn_losses as pinn_losses
import scimba.pinns.pinn_x as pinn_x
import scimba.pinns.training_x as training_x
import scimba.sampling.sampling_parameters as sampling_parameters
import scimba.sampling.sampling_pde as sampling_pde
import scimba.sampling.uniform_sampling as uniform_sampling
from scimba.equations import domain, pdes

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846


class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self, space_domain, form):
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=0,
            parameter_domain=[],
        )

        self.first_derivative = True
        self.second_derivative = True
        self.form = form
        self.sdf = space_domain.large_domain.sdf

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        u = self.get_variables(w)
        return u

    def residual(self, w, x, mu, **kwargs):
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        f = 1.0
        return u_xx + u_yy + f

    def post_processing(self, x, mu, w):
        return self.sdf(x) * w

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        return -1./4.*((x1-self.form.x0)**2+(x2-self.form.y0)**2-self.form.r**2)


class Circle2(domain.SignedDistance):
    def __init__(self):
        super().__init__(dim=2)
        self.bounds = [[0.0,1.0],[0.0,1.0]]
        self.x0, self.y0 = 0.5,0.5
        self.r = np.sqrt(2) / 4.0
        print(self.r)

    def sdf(self, x):
        x1, x2 = x.get_coordinates()
        ones = torch.ones_like(x1)
        res = (x1 - self.x0) ** 2 + (x2 - self.y0) ** 2 - self.r **2
        return res


def test_sdf(sdf, new_training = False, plot = False):
    xdomain = domain.SpaceDomain(2, domain.SignedDistanceBasedDomain(2, sdf.bounds, sdf))

    pde = Poisson_2D(xdomain,sdf)
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = "test.pth"

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    tlayers = [40, 40, 40, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(w_res=1.0)
    optimizers = training_tools.OptimizerData(learning_rate=5e-2, decay=0.99)
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
        trainer.train(epochs=500, n_collocation=5000)

    if plot:
        trainer.plot(20000, reference_solution=True)
    # trainer.plot_derivative_mu(n_visu=20000)
    
    return pinn


def main():
    sdf = Circle2()
    test_sdf(sdf, new_training=True, plot=True)


if __name__ == "__main__":
    main()

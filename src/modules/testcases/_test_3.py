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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846
ELLIPSOID_A = 4 / 3
ELLIPSOID_B = 1 / ELLIPSOID_A




class Poisson_2D(pdes.AbstractPDEx):
    def __init__(self, space_domain):
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=4,
            parameter_domain=[[0.5,0.50001],[0.5,0.50001],[0.03,0.0300001],[0.1,0.10001]],
            #parameter_domain=[[0.4,0.6],[0.4,0.6],[0.05,0.2],[1.0,10]],
        )

        self.first_derivative = True
        self.second_derivative = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        u = self.get_variables(w)
        return u

    def residual(self, w, x, mu, **kwargs):
        x1, x2 = x.get_coordinates()
        c1,c2,sigma,eps = self.get_parameters(mu)
        eps = 10*eps
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        u_xy = self.get_variables(w, "w_xy")
        u = self.get_variables(w)
        a11 = eps*x1*x1+x2*x2
        a12 = (eps-1.0)*x1*x2
        a21 = (eps-1.0)*x1*x2
        a22 = x1*x1+eps*x2*x2
        f = 10*torch.exp(-((x1-c1)**2.0+(x2-c2)**2.0)/(2.0*sigma*sigma))
        ### TOOO DOOOO compute the good  (we change the frequency of the sinus)
        return a11*u_xx + a12*u_xy + a21*u_xy + a22*u_yy + f


def Run_laplacian2D(pde):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = "test_fe3.pth"
    # new_training = False
    new_training = False
    training = True

    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    tlayers = [40, 60, 60, 60, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="tanh")
    pinn = pinn_x.PINNx(network, pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=True, w_res=1.0, w_bc=30.0
    )
    optimizers = training_tools.OptimizerData(learning_rate=1.6e-2, decay=0.99,
                                                switch_to_LBFGS=True,
                                                switch_to_LBFGS_at=1000)
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=15000,
    )

    trainer.train(epochs=5000, n_collocation=15000, n_bc_collocation=5000, n_data=0)

    trainer.plot(50000, random=True)
    # trainer.plot_derivative_mu(n_visu=20000)


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[0.0,1.0], [0.0,1.0]]))
    print(xdomain)
    pde = Poisson_2D(xdomain)

    Run_laplacian2D(pde)
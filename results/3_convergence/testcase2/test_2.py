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
            nb_parameters=2,
            parameter_domain=[[-0.5, 0.500001],[-0.50000, 0.500001]],
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
        mu1,mu2 = self.get_parameters(mu)
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        # f = -torch.exp(-((x1 - mu1)**2 + (x2 - mu2)**2)/2) * (((x1**2 - 2*mu1*x1 + mu1**2 - 5)*torch.sin(2*x1) + (4*mu1 - 4*x1)*torch.cos(2*x1)) * torch.sin(2*x2) + torch.sin(2*x1) * ((x2**2 - 2*mu2*x2 + mu2**2 - 5)*torch.sin(2*x2) + (4*mu2 - 4*x2)*torch.cos(2*x2)))
        
        f = torch.exp(-((x1-mu1)**2.0 - (x2-mu2)**2.0)/2.0) * (16.0*(x1-mu1)*torch.sin(8*x2)*torch.cos(8*x1) - 1.0*(x1-mu1)**2.0*torch.sin(8*x1)*torch.sin(8*x2) + 16.0*(x2-mu2)*torch.sin(8*x1)*torch.cos(8*x2) - 1.0*(x2-mu2)**2.0*torch.sin(8*x1)*torch.sin(8*x2) + 130.0*torch.sin(8*x1)*torch.sin(8*x2))

    def post_processing(self, x, mu, w):
        x1, x2 = x.get_coordinates()
        return (0.5*PI- x1) * (x1+ 0.5*PI) * (x2+0.5*PI) * (0.5*PI - x2) * w

    def reference_solution(self, x, mu):
        x1, x2 = x.get_coordinates()
        mu1, mu2= self.get_parameters(mu)
        g= torch.exp(-((x1-mu1)**2.0 +(x2-mu2)**2.0)/2)
        return g * torch.sin(8*x1) * torch.sin(8*x2)


def Run_laplacian2D(pde, bc_loss_bool=False, w_bc=0, w_res=1.0):
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = "test_fe2.pth"
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
    network = pinn_x.MultiScale_Fourier_x(
        pde=pde, stds=[1.5, 10.0], nb_features=20, layer_sizes=tlayers
    )
    pinn = pinn_x.PINNx(network,pde)
    losses = pinn_losses.PinnLossesData(
        bc_loss_bool=bc_loss_bool, w_res=w_res, w_bc=w_bc
    )
    optimizers = training_tools.OptimizerData(learning_rate=1.7e-2, decay=0.99,
                                                switch_to_LBFGS=True,
                                                switch_to_LBFGS_at=1000)
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        sampler=sampler,
        losses=losses,
        optimizers=optimizers,
        file_name=file_name,
        batch_size=6000,
    )

    if not bc_loss_bool:
        if training:
            trainer.train(epochs=5000, n_collocation=6000, n_data=0)
    else:
        if training:
            trainer.train(
                epochs=12, n_collocation=5000, n_bc_collocation=2000, n_data=0
            )

    trainer.plot(20000, random=True,reference_solution=True)
    # trainer.plot_derivative_mu(n_visu=20000)


if __name__ == "__main__":
    # Laplacien strong Bc on Square with nn
    xdomain = domain.SpaceDomain(2, domain.SquareDomain(2, [[-0.5*PI, 0.5*PI], [-0.5*PI, 0.5*PI]]))
    print(xdomain)
    pde = Poisson_2D(xdomain)

    Run_laplacian2D(pde)
from pathlib import Path

import scimba.pinns.pinn_x as pinn_x
import scimba.pinns.training_x as training_x
import scimba.sampling.sampling_parameters as sampling_parameters
import scimba.sampling.sampling_pde as sampling_pde
import scimba.sampling.uniform_sampling as uniform_sampling
import torch
from scimba.equations import domain, pde_1d_laplacian
import scimba.pinns.pinn_losses as pinn_losses
import scimba.nets.training_tools as training_tools
from scimba.equations import domain, pdes
import time

from testcases.geometry.geometry_3D import Cube
from testcases.problem.problem_3D import TestCase1_3D

current = Path(__file__).parent.parent.parent.parent.parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

torch.set_default_dtype(torch.double)
torch.set_default_device(device)

PI = 3.14159265358979323846

class Poisson_3D(pdes.AbstractPDEx):
    def __init__(self):
        self.problem = TestCase1_3D()
        
        assert isinstance(self.problem.geometry, Cube)
        
        space_domain = domain.SpaceDomain(3,domain.SquareDomain(3, self.problem.geometry.box))
                
        super().__init__(
            nb_unknowns=1,
            space_domain=space_domain,
            nb_parameters=self.problem.nb_parameters,
            parameter_domain=[[0.05,0.0500001],[0.22,0.2200001],[0.1,0.100001]], #[0.05 0.22 0.1 ]
        )

        self.first_derivative = True
        self.second_derivative = True

    def make_data(self, n_data):
        pass

    def bc_residual(self, w, x, mu, **kwargs):
        u = self.get_variables(w)
        return u

    def residual(self, w, x, mu, **kwargs):
        x1, x2, x3 = x.get_coordinates()
        mu1,mu2,mu3 = self.get_parameters(mu)
        u_xx = self.get_variables(w, "w_xx")
        u_yy = self.get_variables(w, "w_yy")
        u_zz = self.get_variables(w, "w_zz")
        # f = 12 * PI**2 * alpha * torch.sin(2 * PI * x1) * torch.sin(2 * PI * x2)* torch.sin(2 * PI * x3)
        f = self.problem.f(torch, [x1, x2, x3], [mu1, mu2, mu3])
        return u_xx + u_yy + u_zz+ f

    def post_processing(self, x, mu, w):
        x1, x2, x3 = x.get_coordinates()
        
        a = self.problem.geometry.box[0][0]
        b = self.problem.geometry.box[0][1]
        a2 = self.problem.geometry.box[1][0]
        b2 = self.problem.geometry.box[1][1]
        a3 = self.problem.geometry.box[2][0]
        b3 = self.problem.geometry.box[2][1]
        
        return (x1-a)*(b-x1)*(x2-a2)*(b2-x2)*(x3-a3)*(b3-x3)*w
        # return x1 * (1 - x1) * x2 * (1 - x2) * x3 * (1 - x3) * w

    def reference_solution(self, x, mu):
        x1, x2, x3 = x.get_coordinates()
        mu1,mu2,mu3 = self.get_parameters(mu)
        return self.problem.u_ex(torch, [x1, x2, x3], [mu1, mu2, mu3])
        # return alpha * torch.sin(2 * PI * x1) * torch.sin(2 * PI * x2) * torch.sin(2 * PI * x3)


def Run_laplacian3D(pde, new_training = False, w_res=1.0):
    tps1 = time.time()
    pde = Poisson_3D()
    x_sampler = sampling_pde.XSampler(pde=pde)
    mu_sampler = sampling_parameters.MuSampler(
        sampler=uniform_sampling.UniformSampling, model=pde
    )
    sampler = sampling_pde.PdeXCartesianSampler(x_sampler, mu_sampler)

    file_name = current / "networks"  / "test_3D" / "test_fe1_v1.pth"
    
    if new_training:
        (
            Path.cwd()
            / Path(training_x.TrainerPINNSpace.FOLDER_FOR_SAVED_NETWORKS)
            / file_name
        ).unlink(missing_ok=True)

    # tlayers = [40, 60, 60, 60, 40]
    tlayers = [40, 40, 40, 40, 40]
    network = pinn_x.MLP_x(pde=pde, layer_sizes=tlayers, activation_type="sine")
    pinn = pinn_x.PINNx(network, pde)

    losses = pinn_losses.PinnLossesData(w_res=w_res)
    optimizers = training_tools.OptimizerData(learning_rate=1.5e-2,decay=0.99)
    trainer = training_x.TrainerPINNSpace(
        pde=pde,
        network=pinn,
        losses =losses,
        optimizers=optimizers,
        sampler=sampler,
        file_name=file_name,
        batch_size=2500,
    )
    if new_training:
        trainer.train(epochs=3000, n_collocation=3000, n_init_collocation=0, n_data=0)

    # trainer.plot(n_visu=50000,reference_solution=True)
    filename = current / "networks" / "test_3D" / "test_fe1_v1.png"
    trainer.plot(20000, random=True,reference_solution=True, filename=filename)
    # trainer.plot_derivative_mu(n_visu=500)
    # trainer.plot_derivative_xmu(n_visu=500)
    tps2 = time.time()
    print(" >>>",tps2 - tps1)
    
    return trainer,pinn

if __name__ == "__main__":
    pde = Poisson_3D()
    Run_laplacian3D(pde,new_training=False)

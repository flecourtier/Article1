import dolfinx
import matplotlib.pyplot as plt
import pyvista
import matplotlib.tri as tri

def plot_mesh(mesh: dolfinx.mesh.Mesh, plot_sol=True, filename = None):
    plt.figure(figsize=(5,5))
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    plt.triplot(tria, color="k")
    if filename is not None:
        pyvista.start_xvfb(wait=0.1)
        plt.savefig(filename)
    if plot_sol:
        plt.show()
    plt.close()
    
def plot_sol(V: dolfinx.fem.functionspace, sol: dolfinx.fem.Function, plot_sol=True, filename = None):
    try:
        cells, types, x = dolfinx.plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = sol.x.array.real
        grid.set_active_scalars("u")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        if filename is not None:
            pyvista.start_xvfb(wait=0.1)
            plotter.screenshot(filename)
        if plot_sol:
            plotter.show()
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution")
        print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
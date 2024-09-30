import numpy as np

from pyop2.mpi import COMM_WORLD

from firedrake import TensorBoxMesh
from firedrake.cython import dmcommon
from firedrake import mesh
from firedrake.petsc import PETSc

@PETSc.Log.EventDecorator()
def OriginBoxMesh(
    nx,
    ny,
    nz,
    Lx,
    Ly,
    Lz,
    originX=0.0,
    originY=0.0,
    originZ=0.0,
    hexahedral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal="default",
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :kwarg hexahedral: (optional), creates hexahedral mesh.
    :kwarg distribution_parameters: options controlling mesh
            distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    * 5: plane z == 0
    * 6: plane z == Lz
    """
    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")
    if hexahedral:
        plex = PETSc.DMPlex().createBoxMesh((nx, ny, nz), lower=(originX, originY, originZ), upper=(originX+Lx, originY+Ly, originZ+Lz), simplex=False, periodic=False, interpolate=True, comm=comm)
        plex.removeLabel(dmcommon.FACE_SETS_LABEL)
        nvert = 4  # num. vertices on faect

        # Apply boundary IDs
        plex.createLabel(dmcommon.FACE_SETS_LABEL)
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        cdim = plex.getCoordinateDim()
        assert cdim == 3
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            xtol = Lx / (2 * nx)
            ytol = Ly / (2 * ny)
            ztol = Lz / (2 * nz)
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if all([abs(face_coords[0 + cdim * i]) < xtol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if all([abs(face_coords[0 + cdim * i] - Lx) < xtol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if all([abs(face_coords[1 + cdim * i]) < ytol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
                if all([abs(face_coords[1 + cdim * i] - Ly) < ytol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
                if all([abs(face_coords[2 + cdim * i]) < ztol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 5)
                if all([abs(face_coords[2 + cdim * i] - Lz) < ztol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 6)
        plex.removeLabel("boundary_faces")
        m = mesh.Mesh(
            plex,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )
        return m
    else:
        xcoords = np.linspace(originX, originX+Lx, nx + 1, dtype=np.double)
        ycoords = np.linspace(originY, originY+Ly, ny + 1, dtype=np.double)
        zcoords = np.linspace(originZ, originZ+Lz, nz + 1, dtype=np.double)
        return TensorBoxMesh(
            xcoords,
            ycoords,
            zcoords,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            diagonal=diagonal,
            comm=comm,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
        )
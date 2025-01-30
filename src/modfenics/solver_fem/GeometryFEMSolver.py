from testcases.geometry.geometry_2D import Donut, Square, Circle
from testcases.geometry.geometry_1D import Line
import dolfin as df
from modfenics.solver_fem.FEMSolver import FEMSolver
import numpy as np
import mshr
import time

###############
# Geometry 1D #
###############

class LineFEMSolver(FEMSolver):
    def _create_mesh(self,nb_vert):
        # check if pb_considered is instance of Line class
        assert isinstance(self.pb_considered.geometry, Line)
        
        start = time.time()
        box = np.array(self.pb_considered.geometry.box)
        mesh = df.IntervalMesh(nb_vert - 1, box[0,0], box[0,1])
        end = time.time()
        
        return mesh, end-start

###############
# Geometry 2D #
###############

class SquareFEMSolver(FEMSolver):    
    def _create_mesh(self,nb_vert):
        # check if pb_considered is instance of Square class
        assert isinstance(self.pb_considered.geometry, Square)
        
        start = time.time()
        box = np.array(self.pb_considered.geometry.box)
        mesh = df.RectangleMesh(df.Point(box[0,0], box[1,0]), df.Point(box[0,1], box[1,1]), nb_vert - 1, nb_vert - 1)
        end = time.time()
        
        return mesh, end-start

# For more complicate geometry, we iterate to find the good caracteristic size of the mesh        
class ComplexFEMSolver(FEMSolver):
    def _generate_mesh_given_size(self, domain, nb_vert):
        case_ex = nb_vert==self.N_ex+1
        self.H_start = self.pb_considered.geometry.H_start
        
        box = np.array(self.pb_considered.geometry.box)
        
        mesh_macro = df.RectangleMesh(df.Point(box[0,0], box[1,0]), df.Point(box[0,1], box[1,1]), nb_vert, nb_vert)
        h_macro = mesh_macro.hmax()
        # print("h_macro = ", h_macro)
        if self.H_start is None or not case_ex:
            H = int(nb_vert/3)
            # print("ici")
        else:
            H = self.H_start
        mesh = mshr.generate_mesh(domain,H)
        h = mesh.hmax()
        while h > h_macro:
            H += 1
            start2 = time.time()
            mesh = mshr.generate_mesh(domain,H)
            end2 = time.time()
            h = mesh.hmax()
            # print("H = ", H, "h = ", h)
        
        return mesh, end2-start2
    
class CircleFEMSolver(ComplexFEMSolver):    
    def _create_mesh(self,nb_vert):
        # check if pb_considered is instance of Square class
        assert isinstance(self.pb_considered.geometry, Circle)
        
        center = self.pb_considered.geometry.center
        radius = self.pb_considered.geometry.radius
        
        start = time.time()
        domain = mshr.Circle(df.Point(center[0],center[1]), radius)
        end = time.time()
        
        mesh, tps2 = self._generate_mesh_given_size(domain, nb_vert)

        tps = end-start + tps2        
        return mesh, tps

# For more complicate geometry, we iterate to find the good caracteristic size of the mesh        
class DonutFEMSolver(ComplexFEMSolver):    
    def _create_mesh(self,nb_vert):
        # check if pb_considered is instance of Donut class
        assert isinstance(self.pb_considered.geometry, Donut)
        
        bigcenter = self.pb_considered.geometry.bigcircle.center
        bigradius = self.pb_considered.geometry.bigcircle.radius
        smallcenter = self.pb_considered.geometry.hole.center
        smallradius = self.pb_considered.geometry.hole.radius
        
        start = time.time()
        bigcircle = mshr.Circle(df.Point(bigcenter[0],bigcenter[1]), bigradius)
        hole = mshr.Circle(df.Point(smallcenter[0],smallcenter[1]), smallradius)
        domain = bigcircle-hole
        end = time.time()
        
        mesh, tps2 = self._generate_mesh_given_size(domain, nb_vert)

        tps = end-start + tps2        
        return mesh, tps
    
###############
# Geometry 3D #
###############

class CubeFEMSolver(FEMSolver):    
    def _create_mesh(self,nb_vert):
        # check if pb_considered is instance of Square class
        assert isinstance(self.pb_considered.geometry, Square)
        
        start = time.time()
        box = np.array(self.pb_considered.geometry.box)
        mesh = df.BoxMesh(df.Point(box[0,0], box[1,0], box[2,0]), df.Point(box[0,1], box[1,1], box[2,1]), nb_vert - 1, nb_vert - 1, nb_vert - 1)
        end = time.time()
        
        return mesh, end-start
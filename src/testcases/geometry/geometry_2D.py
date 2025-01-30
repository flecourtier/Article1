PI = 3.14159265358979323846

class Square:
    def __init__(self,a,b,a2,b2):
        self.box = [[a,b],[a2,b2]]
        self.center = [(a+b)/2,(a2+b2)/2]
        self.radius = (b-a)/2
        
class Square1(Square):
    def __init__(self):
        super().__init__(-0.5*PI, 0.5*PI, -0.5*PI, 0.5*PI)
        
class UnitSquare(Square):
    def __init__(self):
        super().__init__(0.0, 1.0, 0.0, 1.0)
    
class Circle:
    def __init__(self,a,b,r):
        self.center = [a,b]
        self.radius = r
        eps = 0.1
        self.box = [[a-r-eps,a+r+eps],[b-r-eps,b+r+eps]]
        self.H_start = 320
        
class Circle2(Circle):
    def __init__(self):
        super().__init__(0.5, 0.5, 1.0)
        
class UnitCircle(Circle):
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0)
        self.box = [[-1,1],[-1,1]]
        
class Donut:
    def __init__(self,a,b,bigr,smallr):
        self.bigcircle = Circle(a,b,bigr)
        self.hole = Circle(a,b,smallr)
        self.H_start = 350 # To construct more quickly mesh_ex (with FEniCS)
        
class Donut1(Donut):
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0, 0.5)
        self.box = [[-1,1],[-1,1]]
        
class Donut2(Donut):
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0, 0.25)
        self.box = [[-1,1],[-1,1]]
        
class SquareDonut:
    def __init__(self,x0,y0,bigr,smallr):
        self.bigsquare = Square(x0-bigr,x0+bigr,y0-bigr,y0+bigr)
        self.hole = Square(x0-smallr,x0+smallr,y0-smallr,y0+smallr)
        
class SquareDonut1(SquareDonut):
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0, 0.5)
        self.box = [[-1.2,1.2],[-1.2,1.2]]
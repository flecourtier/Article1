PI = 3.14159265358979323846

class Square:
    def __init__(self,a,b,a2,b2):
        self.box = [[a,b],[a2,b2]]
        
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
        self.box = None
        
class UnitCircle(Circle):
    def __init__(self):
        super().__init__(0.0, 0.0, 1.0)
        self.box = [[-1,1],[-1,1]]
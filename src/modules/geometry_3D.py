PI = 3.14159265358979323846

class Cube:
    def __init__(self,a,b,a2,b2,a3,b3):
        self.box = [[a,b],[a2,b2],[a3,b3]]
        
class Cube1(Cube):
    def __init__(self):
        super().__init__(-0.5*PI, 0.5*PI, -0.5*PI, 0.5*PI, -0.5*PI, 0.5*PI)
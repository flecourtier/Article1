PI = 3.14159265358979323846

class Line:
    def __init__(self,a,b):
        self.box = [[a,b]]
        
class Line1(Line):
    def __init__(self):
        super().__init__(0.0,1.0)
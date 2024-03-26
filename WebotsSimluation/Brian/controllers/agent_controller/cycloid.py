from math import sin, cos, pi


class cycloid():
    def __init__(self, S, H, Tm):
        self.S = S
        self.H = H
        self.Tm = Tm

    def generate(self, t):
        if t < self.Tm:
            x = self.S*(t/self.Tm-1/(2*pi)*sin(2*pi*t/self.Tm))
            y = self.H*((1 if t < self.Tm/2 else -1) *
                        (2*(t/self.Tm-1/(4*pi)*sin(4*pi*t/self.Tm))-1)+1)
        else:
            x = self.S*((2*self.Tm-t)/self.Tm+1/(2*pi)*sin(2*pi*t/self.Tm))
            y = 0
        return -x+self.S/2 , -y
from math import sin, cos, sqrt, acos, atan, pi

class kinematic():
    def __init__(self, l1, l2 , da1 , da2):
        self.l1 = l1
        self.l1_2 = l1**2
        self.l2 = l2
        self.l2_2 = l2**2
        self.da1 = da1*pi/180
        self.da2 = da2*pi/180

    def angle2pos(self, a1, a2):
        a2 *= -1
        a1 += self.da1
        a2 += self.da2
        x = self.l1*cos(a1)+self.l2*cos(a1+a2)
        y = self.l1*sin(a1)+self.l2*sin(a1+a2)
        return x, y

    def pos2angle(self, x, y):
        l_2 = x**2+y**2
        l = l_2**0.5
        a2 = pi - acos((self.l1_2+self.l2_2-l_2)/(2*self.l1*self.l2))
        a1 = pi/2 - acos((l_2+self.l1_2-self.l2_2)/(2*l*self.l1)) - atan(x/y)
        a1 -= self.da1
        a2 -= self.da2
        return a1 , -a2
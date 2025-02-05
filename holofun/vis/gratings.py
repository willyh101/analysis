import numpy as np
import scipy.signal as sig
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class SquareWaveGrating:
    def __init__(self, sz, sp_freq, phase, orientation, contrast, window_size=1000, bcol=128):
        self.sz = sz # size of the grating, fraction of the screen
        self.sp_freq = sp_freq # spatial frequency, pixels per cycle
        self.phase = phase
        self.orientation = orientation # degrees
        self.contrast = contrast # 0 to 1
        self.window_size = window_size
        self.bcol = bcol # background color
        
    def mesh(self):
        x = np.linspace(-self.window_size/2, self.window_size/2, self.window_size)
        y = np.linspace(-self.window_size/2, self.window_size/2, self.window_size)
        xx, yy = np.meshgrid(x, y)
        return xx, yy
        
    def calc_gradient(self):
        xx, yy = self.mesh()
        grad = np.sin(self.orientation * np.pi / 180) * xx + np.cos(self.orientation * np.pi / 180) * yy
        return grad
    
    def calc_grating(self, grad):
        grating = sig.square((2 * np.pi * grad) / self.sp_freq + (self.phase * np.pi) / 180)
        grating = self.contrast*(self.bcol-1)*grating + self.bcol
        return grating
    
    def calc_bkgd(self):
        bkgrd = np.ones((self.window_size, self.window_size)) * self.bcol
        return bkgrd
    
    def sq_aperture(self):
        xx, yy = self.mesh()
        apt = np.logical_and(
            xx**2 < (self.window_size*self.sz/2)**2,
            yy**2 < (self.window_size*self.sz/2)**2
        )
        return apt
    
    def circ_aperture(self):
        h, w = self.window_size, self.window_size
        c = (h/2,w/2)
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - c[0])**2 + (Y-c[1])**2)
        radius = min(c[0], c[1], w-c[0], h-c[1])
        apt = dist_from_center <= radius*self.sz
        return apt
                
    def create(self):
        grad = self.calc_gradient()
        grating = self.calc_grating(grad)
        bkgd = self.calc_bkgd()
        apt = self.circ_aperture()
        bkgd[apt]= grating[apt]
        return bkgd
    
class AbstractGrating(ABC):
    
    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def apply_contrast(self):
        pass

class SquareWaveGrating2(AbstractGrating):
    def __init__(self, sz=0.8, sp_freq=50, phase=0, orientation=0, contrast=1):
        self.sz = sz # size of the grating, fraction of the screen
        self.sp_freq = sp_freq # spatial frequency, pixels per cycle
        self.phase = phase
        self.orientation = orientation # degrees
        self.contrast = contrast # 0 to 1
    
    def create(self, xx, yy):
        self.grad = np.sin(self.orientation * np.pi / 180) * xx + np.cos(self.orientation * np.pi / 180) * yy
        self.grating = sig.square((2 * np.pi * self.grad) / self.sp_freq + (self.phase * np.pi) / 180)
    
    def apply_contrast(self, grey_level):
        self.grating = self.contrast*(grey_level-1)*self.grating + grey_level

    def __getitem__(self, key):
        return self.grating[key]

    
class AbstractAperture(ABC):
    
    @abstractmethod
    def create(self):
        pass
    
class CircAperture(AbstractAperture):
    def __init__(self, sz: float):
        self.sz = sz # fraction of the screen
    
    def create(self, window_size=1000):
        h, w = window_size, window_size
        c = (h/2,w/2)
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - c[0])**2 + (Y-c[1])**2)
        radius = min(c[0], c[1], w-c[0], h-c[1])
        apt = dist_from_center <= radius*self.sz
        return apt
    
    def __call__(self, window_size=1000):
        return self.create(window_size)
    
class Window:
    def __init__(self, window_size=1000, grey_level=128):
        self.window_size = window_size
        self.grey_level = grey_level
        self.window = np.ones((self.window_size, self.window_size))*grey_level
        self.xx, self.yy = self.mesh()

    def mesh(self):
        x = np.linspace(-self.window_size/2, self.window_size/2, self.window_size)
        y = np.linspace(-self.window_size/2, self.window_size/2, self.window_size)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def add_grating(self, grating: AbstractGrating):
        self.grating = grating
        self.grating.create(self.xx, self.yy)
        self.grating.apply_contrast(self.grey_level)

    def apply_aperture(self, apt: AbstractAperture, fill=None):
        apt = apt(self.window_size)
        self.window[apt] = self.grating[apt]
        if fill is not None:
            self.window[~apt] = fill[~apt] 

    def display(self):
        plt.figure(figsize=(4,4))
        plt.imshow(self.window, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def get_window(self):
        """Returns a copy of the window"""
        return self.window.copy()

def display_grating(im):
    plt.figure(figsize=(4,4))
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
def test_grating():
    grating = SquareWaveGrating(0.8, 100, 0, 45, 1).create()
    display_grating(grating)

def test_grating2():
    win = Window()
    grating = SquareWaveGrating2(0.8, 100, 0, 45, 1)
    win.add_grating(grating)
    apt = CircAperture(0.8)
    win.apply_aperture(apt)
    win.display()

    
if __name__ == '__main__':
    test_grating2()
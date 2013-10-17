import numpy as np
from . import _pyhough_pywrap
import time

class Hough(dict):
    def __init__(self, image, ntheta=None):
        if image.ndim != 2:
            raise ValueError("Input image must be 2D array")

        self.image = image.astype(np.bool)

        self.drho = 1.0
        rows,cols = self.image.shape
        self.nrho = 2*np.ceil(np.sqrt(np.max(rows**2 + cols**2)) / self.drho) + 1

        if (ntheta is None):
            ntheta = self.calc_ntheta_default()

        self.ntheta = ntheta

        self._pyhough = None

    def calc_ntheta_default(self):
        rows,cols = self.image.shape

        return np.ceil(np.pi*np.sqrt(np.max(rows**2 + cols**2))/self.drho)
    
    def transform(self):
        """
        Something
        """

        self.theta = np.pi*np.arange(self.ntheta)/self.ntheta
        self.rho = self.drho*np.arange(self.nrho)-self.nrho*self.drho/2. + self.drho/2.
        
        self._pyhough = _pyhough_pywrap.Hough(self.image,
                                              self.theta,
                                              self.rho)

        t0=time.clock()
        self.image_trans = self._pyhough.transform()
        print "Runtime: %.2f" % (time.clock() - t0)

        # Return transformed image and values
        return self.image_trans,self.theta,self.rho

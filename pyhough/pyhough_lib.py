import numpy as np
from . import _pyhough_pywrap

class Hough(dict):
    def __init__(self, image, drho=None, nrho=None, ntheta=None):
        if image.ndim != 2:
            raise ValueError("Input image must be 2D array")

        self.image = image.astype(np.bool)
        
        if (drho is None):
            drho = self.calc_drho_default()

        self.drho = drho

        if (nrho is None):
            nrho = self.calc_nrho_default()

        self.nrho = nrho

        if (ntheta is None):
            ntheta = self.calc_ntheta_default()

        self.ntheta = ntheta

        self._pyhough = None

    def calc_drho_default(self):
        dx = 1.0
        dy = 1.0
        return np.sqrt((dx**2 + dy**2)/2.)

    def calc_nrho_default(self):
        rows,cols = self.image.shape

        return 2*np.ceil(np.sqrt(np.max(rows**2 + cols**2)) / self.drho) + 1

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
                                              self.drho,
                                              self.rho)
        
        self.image_trans = self._pyhough.transform()

        # hmmm...
        return self.image_trans,self.theta,self.rho

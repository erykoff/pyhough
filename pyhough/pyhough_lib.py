"""
pyhough

Copyright (C) 2013  Eli Rykoff, SLAC.  erykoff at gmail dot com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""

import numpy as np
from . import _pyhough_pywrap
from . import _pyhoughback_pywrap
import time

class Hough(dict):
    """
    A class to do simple Hough transforms on binary data.

    hough = pyhough.Hough(image,ntheta = None)
    transform = hough.transform()

    transform[0] : the Hough transform of image
    transform[1] : the theta (column) values of the transform
    transform[2] : the rho (row) values of the transform

    Each point in theta/rho space defines a line

    rho = x*cos(theta) + y*sin(theta)

    x, y are defined as 0,0 in the lower-left corner (image coordinates)
    
    """
    
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
        transform = hough.transform()

        transform[0] : the Hough transform of image
        transform[1] : the theta (column) values of the transform
        transform[2] : the rho (row) values of the transform
        
        Each point in theta/rho space defines a line
        
        rho = x*cos(theta) + y*sin(theta)
        
        x, y are defined as 0,0 in the lower-left corner (image coordinates)    
        """

        self.theta = np.pi*np.arange(self.ntheta)/self.ntheta
        self.rho = self.drho*np.arange(self.nrho)-self.nrho*self.drho/2. + self.drho/2.
        
        self._pyhough = _pyhough_pywrap.Hough(self.image,
                                              self.theta,
                                              self.rho)

        #t0=time.clock()
        self.image_trans = self._pyhough.transform()
        #print "Runtime: %.2f" % (time.clock() - t0)

        # Return transformed image and values
        return self.image_trans,self.theta,self.rho

class Back(dict):
    """
    A class to do Hough back projections

    """

    def __init__(self, transform, theta, rho, nx, ny) :
        if transform.ndim != 2:
            raise ValueError("Input transform must be 2D array")

        self.transform = transform.astype(np.uint16)

        self.rho = rho
        self.theta = theta
        self.nx = nx
        self.ny = ny
        self._pyhough_back = None

    def backproject(self):
        """
        image = houghback.backproject()
        """

        self._pyhough_back = _pyhoughback_pywrap.Back(self.transform,
                                                      self.theta,
                                                      self.rho,
                                                      self.nx,
                                                      self.ny)

        self.image = self._pyhough_back.backproject()

        return self.image

    

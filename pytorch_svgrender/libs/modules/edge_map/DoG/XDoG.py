# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import filters


class XDoG:

    def __init__(self,
                 gamma=0.98,
                 phi=200,
                 eps=-0.1,
                 sigma=0.8,
                 k=10,
                 binarize: bool = True):
        """
        XDoG algorithm.

        Args:
            gamma: Control the size of the Gaussian filter
            phi: Control changes in edge strength
            eps: Threshold for controlling edge strength
            sigma: The standard deviation of the Gaussian filter controls the degree of smoothness
            k: Control the size ratio of Gaussian filter, (k=10 or k=1.6)
            binarize(bool): Whether to binarize the output
        """

        super(XDoG, self).__init__()

        self.gamma = gamma
        assert 0 <= self.gamma <= 1

        self.phi = phi
        assert 0 <= self.phi <= 1500

        self.eps = eps
        assert -1 <= self.eps <= 1

        self.sigma = sigma
        assert 0.1 <= self.sigma <= 10

        self.k = k
        assert 1 <= self.k <= 100

        self.binarize = binarize

    def __call__(self, img):
        # to gray if image is not already grayscale
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        if np.isnan(img).any():
            img[np.isnan(img)] = np.mean(img[~np.isnan(img)])

        # gaussian filter
        imf1 = ndi.gaussian_filter(img, self.sigma)
        imf2 = ndi.gaussian_filter(img, self.sigma * self.k)
        imdiff = imf1 - self.gamma * imf2

        # XDoG
        imdiff = (imdiff < self.eps) * 1.0 + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))

        # normalize
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()

        if self.binarize:
            th = filters.threshold_otsu(imdiff)
            imdiff = (imdiff >= th).astype('float32')

        return imdiff

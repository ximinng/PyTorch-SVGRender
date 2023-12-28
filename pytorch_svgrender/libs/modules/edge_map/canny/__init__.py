# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import cv2


class CannyDetector:

    def __call__(self, img, low_threshold, high_threshold, L2gradient=False):
        return cv2.Canny(img, low_threshold, high_threshold, L2gradient)


__all__ = ['CannyDetector']

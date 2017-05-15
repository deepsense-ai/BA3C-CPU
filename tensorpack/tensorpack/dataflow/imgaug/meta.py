#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: meta.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from .base import ImageAugmentor

__all__ = ['RandomChooseAug', 'MapImage', 'Identity', 'RandomApplyAug']

class Identity(ImageAugmentor):
    def _augment(self, img, _):
        return img

class RandomApplyAug(ImageAugmentor):
    """ Randomly apply the augmentor with a prob. Otherwise do nothing"""
    def __init__(self, aug, prob):
        super(RandomApplyAug, self).__init__()
        self._init(locals())

    def _get_augment_params(self, img):
        p = self.rng.rand()
        if p < self.prob:
            prm = self.aug._get_augment_params(img)
            return (True, prm)
        else:
            return (False, None)

    def reset_state(self):
        super(RandomApplyAug, self).reset_state()
        self.aug.reset_state()

    def _augment(self, img, prm):
        if not prm[0]:
            return img
        else:
            return self.aug._augment(img, prm[1])

class RandomChooseAug(ImageAugmentor):
    def __init__(self, aug_lists):
        """
        :param aug_lists: list of augmentor, or list of (augmentor, probability) tuple
        """
        super(RandomChooseAug, self).__init__()
        if isinstance(aug_lists[0], (tuple, list)):
            prob = [k[1] for k in aug_lists]
            aug_lists = [k[0] for k in aug_lists]
            self._init(locals())
        else:
            prob = 1.0 / len(aug_lists)
            self._init(locals())

    def reset_state(self):
        super(RandomChooseAug, self).reset_state()
        for a in self.aug_lists:
            a.reset_state()

    def _get_augment_params(self, img):
        aug_idx = self.rng.choice(len(self.aug_lists), p=self.prob)
        aug_prm = self.aug_lists[aug_idx]._get_augment_params(img)
        return aug_idx, aug_prm

    def _augment(self, img, prm):
        idx, prm = prm
        return self.aug_lists[idx]._augment(img, prm)

class MapImage(ImageAugmentor):
    """
    Map the image array by a function.
    """
    def __init__(self, func):
        """
        :param func: a function which takes a image array and return a augmented one
        """
        self.func = func

    def _augment(self, img, _):
        return self.func(img)


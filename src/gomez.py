#!/usr/bin/env python
# coding: utf-8

"""
    implementation of a varitionnal SWOT filter,
    based on the work by Gomez-Navarro et al. (https://doi.org/10.3390/rs12040734)
    adapted from the SWOTdenoise code @https://github.com/meom-group/SWOTmodule
"""

import numpy as np
from scipy import ndimage
from copy import deepcopy
import matplotlib.pyplot as plt

class VariationnalFilter(object):

    def __init__(self, lon, lat, ssha, mask, debug=False):
        self._lon = lon
        self._lat = lat
        self._ssha = np.ma.array(ssha, mask=mask)
        self._mask = mask.astype(int)
        self._debug = debug

    @property
    def debug(self):
        return self._debug

    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def ssha(self):
        return self._ssha

    @property
    def mask(self):
        return self._mask

    @property
    def invmask(self):
        return 1 - self.mask

    def mapper(self, img, png, title):
        """ convenience mapper for debug """
        f, ax = plt.subplots(1,1)
        ax.imshow(img)
        ax.set_title(title)
        f.savefig(png)
        plt.close(f)

    def filter(self, **kwargs):
        ''' high level filter application '''

        precond_params = kwargs.pop('precondition', 10.)
        filter_params = kwargs.pop('filter_params', (0., 10., 0.))
        epsilon = kwargs.pop('epsilon', 1e-6)
        itermax = kwargs.pop('itermax', 1000)

        # if self.debug:
        #     self.mapper(self.ssha,
        #             '/datalocal/pprandi/SandBox/swot/debug/var/input_ssha.png',
        #             'ssha before precondition')

        ssh_pc = self._precondition(precond_params)

        # print(np.shape(ssh_pc))

        # if self.debug:
        #     self.mapper(ssh_pc,
        #        '/datalocal/pprandi/SandBox/swot/debug/var/precondition.png',
        #        'after precondition')

        ssh_f = self._iterations(self.ssha,
                                 ssh_pc,
                                 filter_params,
                                 epsilon,
                                 itermax)
        return ssh_f

    def _precondition(self, sigma):
        ''' apply a gaussian filter for preconditionning '''

        #print('*precondition')
        v = self.ssha.data.copy()
        #print('** shape(v)', np.shape(v))
        v[self.ssha.mask] = 0.
        indic = np.ones_like(v)
        #print('** shape(indic)', np.shape(indic))
        indic[self.ssha.mask] = 0.

        v[:] = ndimage.gaussian_filter(v, sigma=sigma)
        indic[:] = ndimage.gaussian_filter(indic, sigma=sigma)

        indic = np.clip(indic, 1e-8, 1.)


        precond = v/indic
        #print('** shape(precond)', np.shape(precond))
        return precond

    def _iterations(self, ssh, ssh_pc, param, epsilon, itermax):
        ''' perform iterations '''
        #orig_param = deepcopy(param)
        param_max = max(param)
        scaler = 1.
        if param_max > 1.:
            scaler = scaler/param_max
            param = [ii/param_max for ii in param]

        tau = 1./(scaler
                  + 8*param[0]
                  +64*param[1]
                  +512*param[2])
        #mask = 1 - self.mask
        cssh = ssh_pc.copy()
        t = 1.
        iteration = 0
        norm = []

        while iteration < itermax:
            lap = self.laplacian(cssh)
            bilap = self.laplacian(lap)
            trilap = self.laplacian(bilap)

            incr = self.invmask*(ssh.data-cssh)*scaler + param[0]*lap - param[1]*bilap + param[2]*trilap



            ssh_tmp = cssh + tau*incr

            t0 = t
            t = (1+np.sqrt(1+4*(t0**2))) / 2
            cssh = ssh_tmp + (t0-1.)/t*(ssh_tmp-ssh_pc)

            norm.append(np.ma.max(np.abs(ssh_tmp-ssh_pc)))


            ssh_pc = np.copy(ssh_tmp)
            # if self.debug:
            #     self.mapper(ssh_pc,
            #             '/datalocal/pprandi/SandBox/swot/debug/var/filter_iter_%05d.png' %iteration,
            #             'after iteration %05d' %iteration)

            iteration += 1
            #print('norm: %f' %norm[-1])
            if norm[-1] < epsilon:
                break

        return ssh_pc

    def cost_function(self, mask, data_ref, data, params):
        ''' fonction de coût '''
        raise NotImplementedError()

    def laplacian(self, u):
        ''' calcul du laplacien '''
        lap = self.div(self.gradx(u), self.grady(u))
        return lap

    def gradx(self, u):
        ''' gradient dans la direction x '''
        (m,n) = np.shape(u)
        grad = np.zeros((m,n))
        grad[0:-1,:] = np.subtract(u[1::,:], u[0:-1,:])
        return grad

    def grady(self, u):
        ''' gradient dans la direction y '''
        (m,n) = np.shape(u)
        grad = np.zeros((m,n))
        grad[:,0:-1] = np.subtract(u[:,1::], u[:,0:-1])
        return grad

    def div(self, px, py):
        ''' divergence '''
        m, n = px.shape
        M = np.ma.zeros([m,n])
        Mx = np.ma.zeros([m,n])
        My = np.ma.zeros([m,n])

        Mx[1:m-1, :] = px[1:m-1, :] - px[0:m-2, :]
        Mx[0, :] = px[0, :]
        Mx[m-1, :] = -px[m-2, :]

        My[:, 1:n-1] = py[:, 1:n-1] - py[:, 0:n-2]
        My[:, 0] = py[:,0]
        My[:, n-1] = -py[:, n-2]

        M = Mx + My;
        return M



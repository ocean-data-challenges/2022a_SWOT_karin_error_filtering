import numpy as np
import xarray as xr
import os
#import sys
import glob
import pyinterp
from scipy import signal
from itertools import chain
import hvplot.xarray
import pandas as pd
import warnings
import matplotlib.pylab as plt
#sys.path.append('..')
#from src.swot import *
warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .swot import *


class Benchmark(object):

    def __init__(self, gridstep=1):
        self._gridstep = gridstep
        self._stats = ['ssh','grad_ssh_across','grad_ssh_along','ssh_rmse', 'ug_rmse', 'ksi_rmse', 'grad_ssh_across_rmse','grad_ssh_along_rmse']
        self._d_stats = dict()
        self._init_accumulators()

    def _init_accumulators(self):
        """ creation des accumulateurs """
        self._xaxis = pyinterp.Axis(np.arange(-180,180,self.gridstep), is_circle=True)
        self._yaxis = pyinterp.Axis(np.arange(-90, 90,self.gridstep), is_circle=False)

        for k in self._stats:
            self._d_stats[k] = pyinterp.Binning2D(self._xaxis, self._yaxis)
            self._d_stats[k].clear()
        
    def raz(self):
        """ remise a zero des accumulateurs """
        for k in self._stats:
            self._d_stats[k].clear()
    
    
    def _coriolis_parameter(self, lat):  
        """Compute the Coriolis parameter for the given latitude:
        ``f = 2*omega*sin(lat)``, where omega is the angular velocity 
        of the Earth.
    
        Parameters
        ----------
        lat : array
          Latitude [degrees].
        """
        omega = 7.2921159e-05  # angular velocity of the Earth [rad/s]
        fc = 2*omega*np.sin(lat*np.pi/180.)
        # avoid zero near equator, bound fc by min val as 1.e-8
        fc = np.sign(fc)*np.maximum(np.abs(fc), 1.e-8)
    
        return fc


    def compute_stats(self, l_files, etuvar, l_files_input):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: unfiltered (true) SSH variable name
            etuvar: filtered SSH variable name
        """

        for i, fname in enumerate(l_files):
            
            # Filtered field
            swt = SwotTrack(fname) #._dset
            swt.compute_geos_current(etuvar, 'filtered_geos_current')
            swt.compute_relative_vorticity('filtered_geos_current_x', 'filtered_geos_current_y', 'filtered_ksi')
            
            # Truth
            swt_input = SwotTrack(l_files_input[i])#._dset
            swt_input.compute_geos_current('ssh_true', 'true_geos_current')
            swt_input.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
            # NEED TO CHEK CONSISTENCY BETWEEN Fileterd and thruth if file not sorted
            
            # SSH RMSE
            self.stats_dict['ssh_rmse'].push(
                swt._dset.longitude.values.flatten(),
                swt._dset.latitude.values.flatten(),
                ((swt._dset[etuvar].values - swt_input._dset['ssh_true'].values)**2).flatten(),
                False
            )
            
            # GEOST CURRENT RMSE
            self.stats_dict['ug_rmse'].push(
                swt._dset.longitude.values.flatten(),
                swt._dset.latitude.values.flatten(),
                ((swt._dset['filtered_geos_current'].values - swt_input._dset['true_geos_current'].values)**2).flatten(),
                False
            )
            
            # VORTICITY RMSE
            self.stats_dict['ksi_rmse'].push(
                swt._dset.longitude.values.flatten(),
                swt._dset.latitude.values.flatten(),
                ((swt._dset['filtered_ksi'].values - swt_input._dset['true_ksi'].values)**2).flatten(),
                False
            )
            
    def _compute_grad_diff(self, etu, ref, f_coriolis):
        """ compute differences of gradients """
        mask = np.isnan(etu)
        ref[mask] = np.nan
        
        # swath resolution is 2kmx2km
        dx = 2000 # m
        dy = 2000 # m
        gravity = 9.81
        
    
        ref_gx, ref_gy = gravity/f_coriolis*np.gradient(ref, dx, edge_order=2)
        etu_gx, etu_gy = gravity/f_coriolis*np.gradient(etu, dx, edge_order=2)
    
        delta_x = etu_gx - ref_gx
        delta_y = etu_gy - ref_gy
        return delta_x, delta_y

    def write_stats(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        to_write = xr.Dataset(
            data_vars=dict(
                # ssh_mean=(["lat", "lon"], self.stats_dict['ssh'].variable('mean').T),
                # ssh_variance=(["lat", "lon"], self.stats_dict['ssh'].variable('variance').T),
                # ssh_count=(["lat", "lon"], self.stats_dict['ssh'].variable('count').T),
                # ssh_minimum=(["lat", "lon"],self.stats_dict['ssh'].variable('min').T),
                # ssh_maximum=(["lat", "lon"], self.stats_dict['ssh'].variable('max').T),
                ssh_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ssh_rmse'].variable('mean').T)),
                ug_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ug_rmse'].variable('mean').T)),
                ksi_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ksi_rmse'].variable('mean').T)),
                # grad_ssh_across_mean=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('mean').T),
                # grad_ssh_across_variance=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('variance').T),
                # grad_ssh_across_count=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('count').T),
                # grad_ssh_across_minimum=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('min').T),
                # grad_ssh_across_maximum=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('max').T),
                # grad_ssh_across_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['grad_ssh_across_rmse'].variable('mean').T)),
                # grad_ssh_along_mean=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('mean').T),
                # grad_ssh_along_variance=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('variance').T),
                # grad_ssh_along_count=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('count').T),
                # grad_ssh_along_minimum=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('min').T),
                # grad_ssh_along_maximum=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('max').T),
                # grad_ssh_along_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['grad_ssh_along_rmse'].variable('mean').T)),
            ),
            coords=dict(
                lon=(["lon"], self.stats_dict['ssh'].x),
                lat=(["lat"], self.stats_dict['ssh'].y),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'statistics_of_residuals',
                filter_type=kwargs['filter'] if 'filter' in kwargs else 'None',
            ),
        )
        to_write.to_netcdf(fname)
        
        del self.stats_dict['ssh_rmse']
        del self.stats_dict['ug_rmse']
        del self.stats_dict['ksi_rmse']
        
        if 'filter' in kwargs:
            self.filter_name = kwargs['filter']
        else :
            self.filter_name = 'None'
            
            
        
    def display_stats(self, fname, **kwargs):
        
        ds = xr.open_dataset(fname)
        
  
        plt.figure(figsize=(18, 15))
        # plt.subplots_adjust(left=0.1,
        #             bottom=0.1, 
        #             right=0.9, 
        #             top=0.9, 
        #             wspace=0.4, 
        #             hspace=0.4)
        
        ax = plt.subplot(311, projection=ccrs.PlateCarree())
        vmin = np.nanpercentile(ds.ssh_rmse, 5)
        vmax = np.nanpercentile(ds.ssh_rmse, 95)
        ds.ssh_rmse.plot(x='lon', y='lat', vmin=vmin, vmax=vmax, cmap='Reds', cbar_kwargs={'label': '[m]'}, **kwargs)
        plt.title('RMSE SSH field', fontweight='bold')
        ax.add_feature(cfeature.LAND, zorder=2)
        ax.coastlines(zorder=2)

        ax = plt.subplot(312, projection=ccrs.PlateCarree())
        vmin = np.nanpercentile(ds.ug_rmse, 5)
        vmax = np.nanpercentile(ds.ug_rmse, 95)
        ds.ug_rmse.plot(x='lon', y='lat', vmin=vmin, vmax=vmax, cmap='Reds', cbar_kwargs={'label': '[m.s$^{-1}$]'}, **kwargs)
        plt.title('RMSE GEOSTROPHIC CURRENT field', fontweight='bold')
        ax.add_feature(cfeature.LAND, zorder=2)
        ax.coastlines(zorder=2)
        
        ax = plt.subplot(313, projection=ccrs.PlateCarree())
        vmin = np.nanpercentile(ds.ksi_rmse, 5)
        vmax = np.nanpercentile(ds.ksi_rmse, 95)
        ds.ksi_rmse.plot(x='lon', y='lat', vmin=vmin, vmax=vmax, cmap='Reds', cbar_kwargs={'label': '[]'}, **kwargs)
        plt.title('RMSE RELATIVE VORTICITY field', fontweight='bold')
        ax.add_feature(cfeature.LAND, zorder=2)
        ax.coastlines(zorder=2)

        plt.show()
         
        
    def plot_stats(self, fname ):
        
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ds = xr.open_dataset(fname) 
        lon,lat=np.meshgrid(ds.lon,ds.lat) 
  

        fig = plt.figure(figsize=(18,18))   

        vmin = np.nanpercentile(ds['ssh_rmse'], 5)
        vmax = np.nanpercentile(ds['ssh_rmse'], 95)
        ax3 = fig.add_subplot(3,1,1, projection=ccrs.PlateCarree())        
        plt.scatter(lon,lat,c=ds['ssh_rmse'].values, vmin=0, vmax=vmax, cmap='Reds')
        plt.colorbar()
        ax3.title.set_text('SSH residual RMSE') 
        ax3.set_extent([-100, 45, 5, 69], crs=ccrs.PlateCarree())
        ax3.add_feature(cfeature.LAND) 

        vmin = np.nanpercentile(ds['grad_ssh_across_rmse'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_across_rmse'], 95)
        ax6 = fig.add_subplot(3,1,2, projection=ccrs.PlateCarree())        
        plt.scatter(lon,lat,c=ds['grad_ssh_across_rmse'].values, vmin=0, vmax=vmax, cmap='Reds')
        plt.colorbar()
        ax6.title.set_text('grad_ac SSH residual RMSE') 
        ax6.set_extent([-100, 45, 5, 69], crs=ccrs.PlateCarree())
        ax6.add_feature(cfeature.LAND) 

        vmin = np.nanpercentile(ds['grad_ssh_along_rmse'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_along_rmse'], 95)
        ax9 = fig.add_subplot(3,1,3, projection=ccrs.PlateCarree())        
        plt.scatter(lon,lat,c=ds['grad_ssh_along_rmse'].values, vmin=0, vmax=vmax, cmap='Reds')
        plt.colorbar()
        ax9.title.set_text('grad_al SSH residual RMSE')
        ax9.set_extent([-100, 45, 5, 69], crs=ccrs.PlateCarree())
        ax9.add_feature(cfeature.LAND)
         


        plt.show()
        
        
        
    def compute_stats_by_regime(self, l_files, etuvar, l_files_inputs, regime_type ='all'):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: unfiltered (true) SSH variable name
            etuvar: filtered SSH variable name
        """
        ds = xr.open_mfdataset(l_files, combine='nested', concat_dim='num_lines', parallel=True)
        ds_input = xr.open_mfdataset(l_files_inputs, combine='nested', concat_dim='num_lines', parallel=True)
        
        ds['residual_noise'] = ds[etuvar] - ds_input['ssh_true']
        ds['karin_noise'] = ds_input['ssh_karin'] - ds_input['ssh_true']
        ds['ssh_true'] = ds_input['ssh_true']
        ds['ssh_karin'] = ds_input['ssh_karin']
        swt = SwotTrack(dset=ds)
        swt.compute_geos_current(etuvar, 'filtered_geos_current')
        swt.compute_relative_vorticity('filtered_geos_current_x', 'filtered_geos_current_y', 'filtered_ksi')
        swt.compute_geos_current('ssh_true', 'true_geos_current')
        swt.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
        swt.compute_geos_current('ssh_karin', 'karin_geos_current')
        swt.compute_relative_vorticity('karin_geos_current_x', 'karin_geos_current_y', 'karin_ksi')
        self.num_pixels = ds.num_pixels.values
        del ds
        
        def compute_mean_rmse(ds, msk):
            
            tmp = ((ds*msk)**2).load()
            ds_stat = pyinterp.DescriptiveStatistics(tmp, axis=0)
            rmse_ac = np.sqrt(ds_stat.mean())
            
            del ds_stat
            
            ds_stat = pyinterp.DescriptiveStatistics(tmp)
            rmse = np.sqrt(ds_stat.mean())
            
            del ds_stat
            del tmp
            
            return rmse_ac, rmse
        
        
        # GLOBAL STAT
        print('processing global')
        msk = np.ones(ds_input.mask_coastline_200km.shape)        
        self.rmse_ac_residual_noise_global, self.rmse_residual_noise_global = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_global, self.rmse_karin_noise_global = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_global_ug, self.rmse_residual_noise_global_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_global_ug, self.rmse_karin_noise_global_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_global_ksi, self.rmse_residual_noise_global_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_global_ksi, self.rmse_karin_noise_global_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        # COASTAL (< 200 km) STAT
        print('processing coastal')
        msk = np.ones(ds_input.mask_coastline_200km.shape) 
        msk[ds_input['mask_coastline_200km']==1] = np.nan        
        self.rmse_ac_residual_noise_coastal, self.rmse_residual_noise_coastal = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_coastal, self.rmse_karin_noise_coastal = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_coastal_ug, self.rmse_residual_noise_coastal_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_coastal_ug, self.rmse_karin_noise_coastal_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_coastal_ksi, self.rmse_residual_noise_coastal_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_coastal_ksi, self.rmse_karin_noise_coastal_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        # OFFSHORE (> 200 km) + LOW VARIABILITY ( < 200cm2) STAT 
        print('processing low var')
        msk = ds_input['mask_coastline_200km']*ds_input['mask_ssh_var_over200cm2']
        self.rmse_ac_residual_noise_offshore_lowvar, self.rmse_residual_noise_offshore_lowvar = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_offshore_lowvar, self.rmse_karin_noise_offshore_lowvar = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_offshore_lowvar_ug, self.rmse_residual_noise_offshore_lowvar_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_offshore_lowvar_ug, self.rmse_karin_noise_offshore_lowvar_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_offshore_lowvar_ksi, self.rmse_residual_noise_offshore_lowvar_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_offshore_lowvar_ksi, self.rmse_karin_noise_offshore_lowvar_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        # OFFSHORE (> 200 km) + HIGH VARIABILITY ( < 200cm2) STAT
        print('processing high var')
        msk = np.ones(ds_input.mask_ssh_var_over200cm2.shape) 
        msk[ds_input['mask_ssh_var_over200cm2']==1] = np.nan
        msk = ds_input['mask_coastline_200km']*msk
        self.rmse_ac_residual_noise_offshore_highvar, self.rmse_residual_noise_offshore_highvar = compute_mean_rmse(swt._dset['residual_noise'], msk)
        self.rmse_ac_karin_noise_offshore_highvar, self.rmse_karin_noise_offshore_highvar = compute_mean_rmse(swt._dset['karin_noise'], msk)
        self.rmse_ac_residual_noise_offshore_highvar_ug, self.rmse_residual_noise_offshore_highvar_ug = compute_mean_rmse(swt._dset['filtered_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_karin_noise_offshore_highvar_ug, self.rmse_karin_noise_offshore_highvar_ug = compute_mean_rmse(swt._dset['karin_geos_current'] - swt._dset['true_geos_current'], msk)
        self.rmse_ac_residual_noise_offshore_highvar_ksi, self.rmse_residual_noise_offshore_highvar_ksi = compute_mean_rmse(swt._dset['filtered_ksi'] - swt._dset['true_ksi'], msk)
        self.rmse_ac_karin_noise_offshore_highvar_ksi, self.rmse_karin_noise_offshore_highvar_ksi = compute_mean_rmse(swt._dset['karin_ksi'] - swt._dset['true_ksi'], msk)
        del msk
        
        
        
    def write_stats_by_regime(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        
        to_write = xr.Dataset(
            data_vars=dict(
                rmse_ac_residual_noise_global = (["num_pixels"], self.rmse_ac_residual_noise_global),
                rmse_residual_noise_global = (["x"], self.rmse_residual_noise_global),
                rmse_ac_karin_noise_global = (["num_pixels"], self.rmse_ac_karin_noise_global),
                rmse_karin_noise_global = (["x"], self.rmse_karin_noise_global),
                rmse_ac_residual_noise_global_ug = (["num_pixels"], self.rmse_ac_residual_noise_global_ug),
                rmse_residual_noise_global_ug = (["x"], self.rmse_residual_noise_global_ug),
                rmse_ac_karin_noise_global_ug = (["num_pixels"], self.rmse_ac_karin_noise_global_ug),
                rmse_karin_noise_global_ug = (["x"], self.rmse_karin_noise_global_ug),
                rmse_ac_residual_noise_global_ksi = (["num_pixels"], self.rmse_ac_residual_noise_global_ksi),
                rmse_residual_noise_global_ksi = (["x"], self.rmse_residual_noise_global_ksi),
                rmse_ac_karin_noise_global_ksi = (["num_pixels"], self.rmse_ac_karin_noise_global_ksi),
                rmse_karin_noise_global_ksi = (["x"], self.rmse_karin_noise_global_ksi),
                
                rmse_ac_residual_noise_coastal = (["num_pixels"], self.rmse_ac_residual_noise_coastal),
                rmse_residual_noise_coastal = (["x"], self.rmse_residual_noise_coastal),
                rmse_ac_karin_noise_coastal = (["num_pixels"], self.rmse_ac_karin_noise_coastal),
                rmse_karin_noise_coastal = (["x"], self.rmse_karin_noise_coastal),
                rmse_ac_residual_noise_coastal_ug = (["num_pixels"], self.rmse_ac_residual_noise_coastal_ug),
                rmse_residual_noise_coastal_ug = (["x"], self.rmse_residual_noise_coastal_ug),
                rmse_ac_karin_noise_coastal_ug = (["num_pixels"], self.rmse_ac_karin_noise_coastal_ug),
                rmse_karin_noise_coastal_ug = (["x"], self.rmse_karin_noise_coastal_ug),
                rmse_ac_residual_noise_coastal_ksi = (["num_pixels"], self.rmse_ac_residual_noise_coastal_ksi),
                rmse_residual_noise_coastal_ksi = (["x"], self.rmse_residual_noise_coastal_ksi),
                rmse_ac_karin_noise_coastal_ksi = (["num_pixels"], self.rmse_ac_karin_noise_coastal_ksi),
                rmse_karin_noise_coastal_ksi = (["x"], self.rmse_karin_noise_coastal_ksi),
                
                rmse_ac_residual_noise_offshore_highvar = (["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar),
                rmse_residual_noise_offshore_highvar = (["x"], self.rmse_residual_noise_offshore_highvar),
                rmse_ac_karin_noise_offshore_highvar = (["num_pixels"], self.rmse_ac_karin_noise_offshore_highvar),
                rmse_karin_noise_offshore_highvar = (["x"], self.rmse_karin_noise_offshore_highvar),
                rmse_ac_residual_noise_offshore_highvar_ug = (["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar_ug),
                rmse_residual_noise_offshore_highvar_ug = (["x"], self.rmse_residual_noise_offshore_highvar_ug),
                rmse_ac_karin_noise_offshore_highvar_ug = (["num_pixels"], self.rmse_ac_karin_noise_offshore_highvar_ug),
                rmse_karin_noise_offshore_highvar_ug = (["x"], self.rmse_karin_noise_offshore_highvar_ug),
                rmse_ac_residual_noise_offshore_highvar_ksi = (["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar_ksi),
                rmse_residual_noise_offshore_highvar_ksi = (["x"], self.rmse_residual_noise_offshore_highvar_ksi),
                rmse_ac_karin_noise_offshore_highvar_ksi = (["num_pixels"], self.rmse_ac_karin_noise_offshore_highvar_ksi),
                rmse_karin_noise_offshore_highvar_ksi = (["x"], self.rmse_karin_noise_offshore_highvar_ksi),
                
                rmse_ac_residual_noise_offshore_lowvar = (["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar),
                rmse_residual_noise_offshore_lowvar = (["x"], self.rmse_residual_noise_offshore_lowvar),
                rmse_ac_karin_noise_offshore_lowvar = (["num_pixels"], self.rmse_ac_karin_noise_offshore_lowvar),
                rmse_karin_noise_offshore_lowvar = (["x"], self.rmse_karin_noise_offshore_lowvar),
                rmse_ac_residual_noise_offshore_lowvar_ug = (["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar_ug),
                rmse_residual_noise_offshore_lowvar_ug = (["x"], self.rmse_residual_noise_offshore_lowvar_ug),
                rmse_ac_karin_noise_offshore_lowvar_ug = (["num_pixels"], self.rmse_ac_karin_noise_offshore_lowvar_ug),
                rmse_karin_noise_offshore_lowvar_ug = (["x"], self.rmse_karin_noise_offshore_lowvar_ug),
                rmse_ac_residual_noise_offshore_lowvar_ksi = (["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar_ksi),
                rmse_residual_noise_offshore_lowvar_ksi = (["x"], self.rmse_residual_noise_offshore_lowvar_ksi),
                rmse_ac_karin_noise_offshore_lowvar_ksi = (["num_pixels"], self.rmse_ac_karin_noise_offshore_lowvar_ksi),
                rmse_karin_noise_offshore_lowvar_ksi = (["x"], self.rmse_karin_noise_offshore_lowvar_ksi),
            
            ),
            coords=dict(
                num_pixels=(["num_pixels"], self.num_pixels),
                x=(["x"], [1]),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'statistics_of_residuals_by_regime',
                filter_type=kwargs['filter'] if 'filter' in kwargs else 'None',
            ),
        )
        to_write.to_netcdf(fname)


    def plot_stats_by_regime(self, fname): 

        ds = xr.open_dataset(fname).drop('x')  

        fig = plt.figure(figsize=(12, 12))
        
        plt.subplot(221)
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_global'].values,'b', label='residual_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_coastal'].values,'r', label='residual_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_lowvar'].values,'g', label='residual_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_highvar'].values,'k', label='residual_noise_offshore_highvar')


        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_global'].values,'b--', lw=2, label='karin_noise_global')
        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_coastal'].values,'r--', lw=2, label='karin_noise_coastal')
        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_offshore_lowvar'].values,'g--', lw=2, label='karin_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70,ds['rmse_ac_karin_noise_offshore_highvar'].values,'k--', lw=2, label='karin_noise_offshore_highvar')
        
        plt.ylim(0, 0.03)
        plt.xlabel('Across-track distance [km]', fontweight='bold')
        plt.ylabel('RMSE [m]', fontweight='bold')
        plt.title('SEA SURFACE HEIGHT', fontweight='bold')
        
        plt.grid()
        #plt.axis([0,72,0,0.03])
        
        plt.subplot(222)
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_global_ug'].values,'b', label='residual_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_coastal_ug'].values,'r', label='residual_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_lowvar_ug'].values,'g', label='residual_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_highvar_ug'].values,'k', label='residual_noise_offshore_highvar')

        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global_ug'].values,'b--', label='karin_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal_ug'].values,'r--', label='karin_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar_ug'].values,'g--', label='karin_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar_ug'].values,'k--', label='karin_noise_offshore_highvar')
        #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncol=2)
        plt.title('GEOSTROPHIC CURRENTS', fontweight='bold')
        plt.xlabel('Across-track distance [km]', fontweight='bold')
        plt.ylabel('RMSE [m.s$^{-1}$]', fontweight='bold')
        plt.grid()
        
        
        plt.subplot(223)
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_global_ksi'].values,'b', label='residual_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_coastal_ksi'].values,'r', label='residual_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_lowvar_ksi'].values,'g', label='residual_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_residual_noise_offshore_highvar_ksi'].values,'k', label='residual_noise_offshore_highvar')

        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_global_ksi'].values,'b--', label='karin_noise_global')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_coastal_ksi'].values,'r--', label='karin_noise_coastal')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_lowvar_ksi'].values,'g--', label='karin_noise_offshore_lowvar')
        plt.plot(2*ds.num_pixels - 70, ds['rmse_ac_karin_noise_offshore_highvar_ksi'].values,'k--', label='karin_noise_offshore_highvar')
        #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncol=2)
        plt.title('RELATIVE VORTIVITY', fontweight='bold')
        plt.xlabel('Across-track distance [km]', fontweight='bold')
        plt.ylabel('RMSE []', fontweight='bold')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.02, 0.5, 1, 0.2), loc="upper left", ncol=2)
        
        
        
        
        
        
        
        
  
    
        
    
    def display_stats_by_regime(self, fname):
        
        ds = xr.open_dataset(fname).drop('x')
        
        fig1 = ds.std_ac_residual_noise_global.hvplot.line(x='num_pixels', ylim=(0, 0.025), label='residual_noise_global') * ds.std_ac_karin_noise_global.hvplot.line(x='num_pixels', label='karin_noise_global')
        fig2 = ds.std_ac_residual_noise_coastal.hvplot.line(x='num_pixels', ylim=(0, 0.025),label='residual_noise_coastal') * ds.std_ac_karin_noise_coastal.hvplot.line(x='num_pixels',label='karin_noise_coastal')
        fig3 = ds.std_ac_residual_noise_offshore_lowvar.hvplot.line(x='num_pixels', ylim=(0, 0.025), label='residual_noise_offshore_lowvar') * ds.std_ac_karin_noise_offshore_lowvar.hvplot.line(x='num_pixels', label='karin_noise_offshore_lowvar')
        fig4 = ds.std_ac_residual_noise_offshore_highvar.hvplot.line(x='num_pixels', ylim=(0, 0.025), label='residual_noise_offshore_highvar') * ds.std_ac_karin_noise_offshore_highvar.hvplot.line(x='num_pixels', label='karin_noise_offshore_highvar')
        
        return (fig1* fig2 *fig3 *fig4).opts(legend_position='right',frame_height=300, frame_width=500) 
    
          
    
    def compute_along_track_psd(self, l_files, etuvar, l_files_inputs, lengh_scale=512, overlay=0.25, details=False):
        """ compute along track psd """
            
        
        def create_segments_from_1d(lon, lat, ssh_true, ssh_noisy, ssh_filtered, npt=512, n_overlay=128, center=True):
            """
            decoupage en segments d'une serie lon,lat,ssh 1D
            on suppose que les lon/lat sont toutes définies, mais les ssh peuvent avoir des trous
            """
            l_segments_ssh_true = []
            l_segments_ssh_noisy = []
            l_segments_ssh_filtered = []
    
            # parcours des données
            n_obs = len(lon)
            ii=0
            while ii+npt < n_obs:
                seg_ssh_true = ssh_true[ii:ii+npt]
                seg_ssh_noisy = ssh_noisy[ii:ii+npt]
                seg_ssh_filtered = ssh_filtered[ii:ii+npt]
                if (not np.any(np.isnan(seg_ssh_true))) and (not np.any(np.isnan(seg_ssh_noisy))) and (not np.any(np.isnan(seg_ssh_filtered))):
                
                    l_segments_ssh_true.append(seg_ssh_true)
                    l_segments_ssh_noisy.append(seg_ssh_noisy)
                    l_segments_ssh_filtered.append(seg_ssh_filtered)
                
                # l_segments.append(
                #     Segment(
                #         lon[ii:ii+l],
                #         lat[ii:ii+l],
                #         seg_ssh-np.mean(seg_ssh)
                #     )
                # )
                ii+=npt-n_overlay
            return l_segments_ssh_true, l_segments_ssh_noisy, l_segments_ssh_filtered
        
        l_segment_ssh_true = []
        l_segment_ssh_noisy = []
        l_segment_ssh_filtered = []
        
        l_segment_ug_true = []
        l_segment_ug_noisy = []
        l_segment_ug_filtered = []
        
        l_segment_ksi_true = []
        l_segment_ksi_noisy = []
        l_segment_ksi_filtered = []
        
        resolution = 2. # along-track resolution in km
        npt = int(lengh_scale/resolution)
        n_overlap = int(npt*overlay)
        
        
        for i, fname in enumerate(l_files):
            #swt = SwotTrack(fname)
            
            # Filtered field
            swt = SwotTrack(fname) #._dset
            swt.compute_geos_current(etuvar, 'filtered_geos_current')
            swt.compute_relative_vorticity('filtered_geos_current_x', 'filtered_geos_current_y', 'filtered_ksi')
            
            # Truth
            swt_input = SwotTrack(l_files_inputs[i])#._dset
            swt_input.compute_geos_current('ssh_true', 'true_geos_current')
            swt_input.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
            
            # 
            swt_input.compute_geos_current('ssh_karin', 'simulated_noise_geos_current')
            swt_input.compute_relative_vorticity('simulated_noise_geos_current_x', 'simulated_noise_geos_current_y', 'simulated_noise_ksi')
            # NEED TO CHEK CONSISTENCY BETWEEN Fileterd and thruth if file not sorted
            
            
            
            #swt = xr.open_dataset(fname)
            #swt_input = xr.open_dataset(l_files_inputs[i])
        
            # parcours des différentes lignes along-track
            for ac_index in swt._dset.num_pixels.values:
                # extraction des lon/lat/ssh
                lon = swt._dset.longitude.values[:,ac_index]
                lat = swt._dset.longitude.values[:,ac_index]
                
                ssh_true = swt_input._dset['ssh_true'].values[:,ac_index]
                ssh_noisy = swt_input._dset['ssh_karin'].values[:,ac_index]
                ssh_filtered = swt._dset[etuvar].values[:,ac_index]
                
                ug_true = swt_input._dset['true_geos_current'].values[:,ac_index]
                ug_noisy = swt_input._dset['simulated_noise_geos_current'].values[:,ac_index]
                ug_filtered = swt._dset['filtered_geos_current'].values[:,ac_index]
                
                ksi_true = swt_input._dset['true_ksi'].values[:,ac_index]
                ksi_noisy = swt_input._dset['simulated_noise_ksi'].values[:,ac_index]
                ksi_filtered = swt._dset['filtered_ksi'].values[:,ac_index]
            
                # construction de la liste des segments
                al_seg_ssh_true, al_seg_ssh_noisy, al_seg_ssh_filtered = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ssh_true, 
                                                                                                 ssh_noisy, 
                                                                                                 ssh_filtered, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                
                al_seg_ug_true, al_seg_ug_noisy, al_seg_ug_filtered = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ug_true, 
                                                                                                 ug_noisy, 
                                                                                                 ug_filtered, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                    
                    
                al_seg_ksi_true, al_seg_ksi_noisy, al_seg_ksi_filtered = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ksi_true, 
                                                                                                 ksi_noisy, 
                                                                                                 ksi_filtered, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                l_segment_ssh_true.append(al_seg_ssh_true)
                l_segment_ssh_noisy.append(al_seg_ssh_noisy)
                l_segment_ssh_filtered.append(al_seg_ssh_filtered)
                
                l_segment_ug_true.append(al_seg_ug_true)
                l_segment_ug_noisy.append(al_seg_ug_noisy)
                l_segment_ug_filtered.append(al_seg_ug_filtered)
                
                l_segment_ksi_true.append(al_seg_ksi_true)
                l_segment_ksi_noisy.append(al_seg_ksi_noisy)
                l_segment_ksi_filtered.append(al_seg_ksi_filtered)
        
        # on met la liste à plat
        l_flat_ssh_true = np.asarray(list(chain.from_iterable(l_segment_ssh_true))).flatten()
        l_flat_ssh_noisy = np.asarray(list(chain.from_iterable(l_segment_ssh_noisy))).flatten()
        l_flat_ssh_filtered = np.asarray(list(chain.from_iterable(l_segment_ssh_filtered))).flatten()
        
        l_flat_ug_true = np.asarray(list(chain.from_iterable(l_segment_ug_true))).flatten()
        l_flat_ug_noisy = np.asarray(list(chain.from_iterable(l_segment_ug_noisy))).flatten()
        l_flat_ug_filtered = np.asarray(list(chain.from_iterable(l_segment_ug_filtered))).flatten()
        
        l_flat_ksi_true = np.asarray(list(chain.from_iterable(l_segment_ksi_true))).flatten()
        l_flat_ksi_noisy = np.asarray(list(chain.from_iterable(l_segment_ksi_noisy))).flatten()
        l_flat_ksi_filtered = np.asarray(list(chain.from_iterable(l_segment_ksi_filtered))).flatten()
        
        # PSD 
        freq, cross_spectrum = signal.csd(l_flat_ssh_noisy,l_flat_ssh_filtered, fs=1./resolution, nperseg=npt, noverlap=0)
        freq, psd_ssh_true  = signal.welch(l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ssh_noisy = signal.welch(l_flat_ssh_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ssh_filtered = signal.welch(l_flat_ssh_filtered, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err = signal.welch(l_flat_ssh_filtered - l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err_karin = signal.welch(l_flat_ssh_noisy - l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        
        freq, psd_ug_true  = signal.welch(l_flat_ug_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ug_noisy = signal.welch(l_flat_ug_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ug_filtered = signal.welch(l_flat_ug_filtered, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err_ug = signal.welch(l_flat_ug_filtered - l_flat_ug_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err_karin_ug = signal.welch(l_flat_ug_noisy - l_flat_ug_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        
        freq, psd_ksi_true  = signal.welch(l_flat_ksi_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ksi_noisy = signal.welch(l_flat_ksi_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ksi_filtered = signal.welch(l_flat_ksi_filtered, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err_ksi = signal.welch(l_flat_ksi_filtered - l_flat_ksi_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err_karin_ksi = signal.welch(l_flat_ksi_noisy - l_flat_ksi_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        
        self.freq = freq
        self.cross_spectrum = cross_spectrum
        self.psd_ssh_true = psd_ssh_true
        self.psd_ssh_noisy = psd_ssh_noisy
        self.psd_ssh_filtered = psd_ssh_filtered
        self.psd_err = psd_err
        self.psd_err_karin = psd_err_karin
        
        self.psd_ug_true = psd_ug_true
        self.psd_ug_noisy = psd_ug_noisy
        self.psd_ug_filtered = psd_ug_filtered
        self.psd_err_ug = psd_err_ug
        self.psd_err_karin_ug = psd_err_karin_ug
        
        self.psd_ksi_true = psd_ksi_true
        self.psd_ksi_noisy = psd_ksi_noisy
        self.psd_ksi_filtered = psd_ksi_filtered
        self.psd_err_ksi = psd_err_ksi
        self.psd_err_karin_ksi = psd_err_karin_ksi
    
    
    def write_along_track_psd(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        
        def compute_snr1(array, wavenumber, threshold=1.0):
            """
            :param array:
            :param wavenumber:
            :param threshold:
            :return:
            """

            flag_multiple_crossing = False

            zero_crossings =  np.where(np.diff(np.sign(array - threshold)) != 0.)[0]
            if len(zero_crossings) > 1:
                #print('Multiple crossing', len(zero_crossings))
                flag_multiple_crossing = True
                
            list_of_res = []
            if len(zero_crossings) > 0:    
                for index in range(zero_crossings.size):
            
                    if zero_crossings[index] + 1 < array.size:

                        array1 = array[zero_crossings[index]] - threshold
                        array2 = array[zero_crossings[index] + 1] - threshold
                        dist1 = np.log(wavenumber[zero_crossings[index]])
                        dist2 = np.log(wavenumber[zero_crossings[index] + 1])
                        log_wavenumber_crossing = dist1 - array1 * (dist1 - dist2) / (array1 - array2)
                        
                        resolution_scale = 1. / np.exp(log_wavenumber_crossing)

                    else:
                        resolution_scale = 0.
            
                    list_of_res.append(resolution_scale)

            else:
                resolution_scale = 0.
            #print(list_of_res)
            if len(list_of_res) > 0:
                resolution_scale = np.nanmax(np.asarray(list_of_res))
            else:
                resolution_scale = 1000.
        
            return resolution_scale#, flag_multiple_crossing
        
        self.wavelength_snr1_filter = compute_snr1(self.psd_err/self.psd_ssh_true, self.freq)
        self.wavelength_snr1_nofilter = compute_snr1(self.psd_err_karin/self.psd_ssh_true, self.freq)
        
        self.wavelength_snr1_filter_ug = compute_snr1(self.psd_err_ug/self.psd_ug_true, self.freq)
        self.wavelength_snr1_nofilter_ug = compute_snr1(self.psd_err_karin_ug/self.psd_ug_true, self.freq)
        
        self.wavelength_snr1_filter_ksi = compute_snr1(self.psd_err_ksi/self.psd_ksi_true, self.freq)
        self.wavelength_snr1_nofilter_ksi = compute_snr1(self.psd_err_karin_ksi/self.psd_ksi_true, self.freq)
        
        to_write = xr.Dataset(
            data_vars=dict(
                psd_ssh_true=(["wavenumber"], self.psd_ssh_true),
                cross_spectrum_r=(["wavenumber"], np.real(self.cross_spectrum)),
                cross_spectrum_i=(["wavenumber"], np.imag(self.cross_spectrum)),
                psd_ssh_noisy=(["wavenumber"], self.psd_ssh_noisy),
                psd_ssh_filtered=(["wavenumber"], self.psd_ssh_filtered),
                psd_err=(["wavenumber"], self.psd_err),
                psd_err_karin=(["wavenumber"], self.psd_err_karin),
                snr1_filter=(["wavelength_snr1"], [1]),
                
                psd_ug_true=(["wavenumber"], self.psd_ug_true),
                psd_ug_noisy=(["wavenumber"], self.psd_ug_noisy),
                psd_ug_filtered=(["wavenumber"], self.psd_ug_filtered),
                psd_err_ug=(["wavenumber"], self.psd_err_ug),
                psd_err_karin_ug=(["wavenumber"], self.psd_err_karin_ug),
                
                psd_ksi_true=(["wavenumber"], self.psd_ksi_true),
                psd_ksi_noisy=(["wavenumber"], self.psd_ksi_noisy),
                psd_ksi_filtered=(["wavenumber"], self.psd_ksi_filtered),
                psd_err_ksi=(["wavenumber"], self.psd_err_ksi),
                psd_err_karin_ksi=(["wavenumber"], self.psd_err_karin_ksi),

            ),
            coords=dict(
                wavenumber=(["wavenumber"], self.freq),
                wavelength_snr1_filter=(["wavelength_snr1"], [self.wavelength_snr1_filter]),
                wavelength_snr1_filter_ug=(["wavelength_snr1_ug"], [self.wavelength_snr1_filter_ug]),
                wavelength_snr1_filter_ksi=(["wavelength_snr1_ksi"], [self.wavelength_snr1_filter_ksi]),
                wavelength_snr1_nofilter=(["wavelength_snr1"], [self.wavelength_snr1_nofilter]),
                wavelength_snr1_nofilter_ug=(["wavelength_snr1_ug"], [self.wavelength_snr1_nofilter_ug]),
                wavelength_snr1_nofilter_ksi=(["wavelength_snr1_ksi"], [self.wavelength_snr1_nofilter_ksi]),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'PSD analysis',
                filter_type=kwargs['filter'] if 'filter' in kwargs else 'None',
            ),
        )
        
        to_write.to_netcdf(fname)
        
    def display_psd(self, fname):
        
        ds = xr.open_dataset(fname)

        ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})

        fig = plt.figure(figsize=(15, 18))

        ax = plt.subplot(321)
        ds['psd_ssh_true'].plot(x='wavelength', label='PSD(SSH$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
        ds['psd_ssh_noisy'].plot(x='wavelength', label='PSD(SSH$_{noisy}$)', color='r', lw=2)
        ds['psd_ssh_filtered'].plot(x='wavelength', label='PSD(SSH$_{filtered}$)', color='b', lw=2)
        ds['psd_err'].plot(x='wavelength', label='PSD(SSH$_{err}$)', color='grey', lw=2)
        plt.grid(which='both')
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [m.cy$^{-1}$.km$^{-1}$]')
        ax.invert_xaxis()
        plt.title('PSD Sea Surface Height')

        ds['SNR_filter'] = ds['psd_err']/ds['psd_ssh_true']
        ds['SNR_nofilter'] = ds['psd_err_karin']/ds['psd_ssh_true']
        ax = plt.subplot(322)
        ds['SNR_filter'].plot(x='wavelength', label='PSD(SSH$_{err}$)/PSD(SSH$_{true}$)', color='b', xscale='log', lw=3)
        ds['SNR_nofilter'].plot(x='wavelength', label='PSD(Karin$_{noise}$)/PSD(SSH$_{true}$)', color='r', lw=2)
        (ds['SNR_filter']/ds['SNR_filter']).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
        plt.scatter(ds.wavelength_snr1_filter, 1., color='b', zorder=4, label="SNR1 AFTER filter")
        plt.scatter(ds.wavelength_snr1_nofilter, 1., color='r', zorder=4, label="SNR1 BEFORE filter")
        plt.grid(which='both')
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('SNR')

        plt.ylim(0, 2)
        ax.invert_xaxis()
        plt.title('SNR Sea Surface Height')


        ax = plt.subplot(323)
        ds['psd_ug_true'].plot(x='wavelength', label='PSD(Ug$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
        ds['psd_ug_noisy'].plot(x='wavelength', label='PSD(Ug$_{noisy}$)', color='r', lw=2)
        ds['psd_ug_filtered'].plot(x='wavelength', label='PSD(Ug$_{filtered}$)', color='b', lw=2)
        ds['psd_err_ug'].plot(x='wavelength', label='PSD(err)', color='grey', lw=2)
        plt.grid(which='both')
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [m.s$^{-1}$.cy$^{-1}$.km$^{-1}$]')
        ax.invert_xaxis()
        plt.title('PSD Geostrophic current')

        ds['SNR_filter_ug'] = ds['psd_err_ug']/ds['psd_ug_true']
        ds['SNR_nofilter_ug'] = ds['psd_err_karin_ug']/ds['psd_ug_true']
        ax = plt.subplot(324)
        ds['SNR_filter_ug'].plot(x='wavelength', label='PSD(Ug$_{err}$)/PSD(Ug$_{true}$)', color='b', xscale='log', lw=3)
        ds['SNR_nofilter_ug'].plot(x='wavelength', label='PSD(Ug$_{noise}$)/PSD(Ug$_{true}$)', color='r', lw=2)
        (ds['SNR_filter_ug']/ds['SNR_filter_ug']).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
        plt.scatter(ds.wavelength_snr1_filter_ug, 1., color='b', zorder=4, label="SNR1 AFTER filter")
        plt.scatter(ds.wavelength_snr1_nofilter_ug, 1., color='r', zorder=4, label="SNR1 BEFORE filter")
        plt.grid(which='both')
        plt.legend()
        plt.ylim(0, 2)
        ax.invert_xaxis()
        plt.title('SNR Geostrophic current')
        plt.xlabel('wavelenght [km]')
        plt.ylabel('SNR')


        ax = plt.subplot(325)
        ds['psd_ksi_true'].plot(x='wavelength', label='PSD($\zeta_{true}$)', color='k', xscale='log', yscale='log', lw=3)
        ds['psd_ksi_noisy'].plot(x='wavelength', label='PSD($\zeta_{noisy}$)', color='r', lw=2)
        ds['psd_ksi_filtered'].plot(x='wavelength', label='PSD($\zeta_{filtered}$)', color='b', lw=2)
        ds['psd_err_ksi'].plot(x='wavelength', label='psd_err', color='grey', lw=2)
        plt.grid(which='both')
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [s$^{-1}$.cy$^{-1}$.km$^{-1}$]')
        ax.invert_xaxis()
        plt.title('PSD Relative vorticity')

        ds['SNR_filter_ksi'] = ds['psd_err_ksi']/ds['psd_ksi_true']
        ds['SNR_nofilter_ksi'] = ds['psd_err_karin_ksi']/ds['psd_ksi_true']
        ax = plt.subplot(326)
        ds['SNR_filter_ksi'].plot(x='wavelength', label='PSD($\zeta_{err}$)/PSD($\zeta_{true}$)', color='b', xscale='log', lw=3)
        ds['SNR_nofilter_ksi'].plot(x='wavelength', label='PSD($\zeta_{noise}$)/PSD($\zeta_{true}$)', color='r', lw=2)
        (ds['SNR_filter_ksi']/ds['SNR_filter_ksi']).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
        plt.scatter(ds.wavelength_snr1_filter_ksi, 1., color='b', zorder=4, label="SNR1 AFTER filter")
        plt.scatter(ds.wavelength_snr1_nofilter_ksi, 1., color='r', zorder=4, label="SNR1 BEFORE filter")
        plt.grid(which='both')
        plt.legend()
        plt.ylim(0, 2)
        ax.invert_xaxis()
        plt.title('SNR Relative vorticity')
        plt.xlabel('wavelenght [km]')
        plt.ylabel('SNR')
        
        plt.show()
        
     
    
    def plot_psd(self, fname):
        
        ds = xr.open_dataset(fname)
        
        ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})

        plt.figure(figsize=(12,6))    
        plt.subplot(121)
        plt.loglog(ds.wavelength.values,ds['psd_ssh_true'].values, label='psd_ssh_true')
        plt.loglog(ds.wavelength.values,ds['psd_ssh_noisy'].values, label='psd_ssh_noisy')
        plt.loglog(ds.wavelength.values,ds['psd_ssh_filtered'].values, label='psd_ssh_filtered')
        plt.loglog(ds.wavelength.values,ds['psd_err'].values, label='psd_err') 
        plt.legend()
        plt.xlabel('wavelenght [km]')
        plt.ylabel('PSD [m2/cy/km]')
        x=ds.wavelength.values[np.isfinite(ds.wavelength.values)]
        plt.xlim(max(x), min(x))

        plt.subplot(122)
        ds['SNR_filter'] = ds['psd_err']/ds['psd_ssh_true']
        ds['SNR_nofilter'] = ds['psd_err_karin']/ds['psd_ssh_true']
        plt.semilogx(ds.wavelength.values,ds['SNR_filter'].values,c='#2ca02c', label='SNR_filter')
        plt.semilogx(ds.wavelength.values,ds['SNR_nofilter'].values,c='#ff7f0e', label='SNR_nofilter')
        plt.semilogx(ds.wavelength.values,np.ones_like(ds.wavelength.values),c='grey',linestyle='--', label='SNR = 1') 
        plt.legend()
        plt.ylim(0,2)
        plt.xlabel('wavelenght [km]')
        plt.ylabel('Signal-to-Noise Ratio')
        x=ds.wavelength.values[np.isfinite(ds.wavelength.values)]
        plt.xlim(max(x), min(x))

        plt.show()
    
    
    def display_psd_v0(self, fname):
        ds = xr.open_dataset(fname)
    
        ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})
    
        fig1 = ds['psd_ssh_true'].hvplot.line(x='wavelength', label='psd_ssh_true', loglog=True, flip_xaxis=True, grid=True, legend=True, line_width=4, line_color='k')*\
        ds['psd_ssh_noisy'].hvplot.line(x='wavelength', label='psd_ssh_noisy', line_color='r', line_width=3)*\
        ds['psd_ssh_filtered'].hvplot.line(x='wavelength', label='psd_ssh_filtered', line_color='b', line_width=3)*\
        ds['psd_err'].hvplot.line(x='wavelength', label='psd_err', line_color='grey', line_width=3).opts(title='PSD', frame_height=500, frame_width=400, xlabel='wavelength [km]', ylabel='PSD [m2/cy/km]')
        
        ds['Transfer_function'] = np.sqrt(ds.cross_spectrum_r**2 + ds.cross_spectrum_i**2)/ds.psd_ssh_noisy
        fig2 = (ds['Transfer_function'].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, ylim=(0., 1), grid=True, label='Transfer function', legend=True, line_width=3, color='b')*\
                (0.5*ds['Transfer_function']/ds['Transfer_function']).hvplot.line(x='wavelength', logx=True, flip_xaxis=True, label='Tf=0.5', legend=True, line_width=1, line_color='r')).opts(title='Transfer function', frame_height=500, frame_width=400, xlabel='wavelength [km]', ylabel='Transfer function')
        
        ds['SNR_filter'] = ds['psd_err']/ds['psd_ssh_true']
        ds['SNR_nofilter'] = ds['psd_err_karin']/ds['psd_ssh_true']
        fig3 = (ds['SNR_filter'].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, ylim=(-0.1, 2), grid=True, label='SNR filter', legend=True, line_width=3, color='b')*\
                ds['SNR_nofilter'].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, grid=True, label='SNR nofilter', legend=True, line_width=3, color='r')*\
        (ds['SNR_filter']/ds['SNR_filter']).hvplot.line(x='wavelength', logx=True, flip_xaxis=True, label='SNR=1', legend=True, line_width=1, line_color='k')*\
                                         ds.hvplot.scatter(x='wavelength_snr1_filter', y='snr1_filter', color='k')*\
               ds.hvplot.scatter(x='wavelength_snr1_nofilter', y='snr1_filter', color='k')).opts(title='SNR', frame_height=500, frame_width=400, xlabel='wavelength [km]', ylabel='SNR')
    
    
        return (fig1 + fig2 + fig3).cols(3)
    
    
    def summary(self, notebook_name):
        
        wavelength_snr1_filter = self.wavelength_snr1_filter
        wavelength_snr1_nofilter = self.wavelength_snr1_nofilter
        
        rmse_residual_noise_global = self.rmse_residual_noise_global
        rmse_residual_noise_coastal = self.rmse_residual_noise_coastal
        rmse_residual_noise_offshore_lowvar = self.rmse_residual_noise_offshore_lowvar
        rmse_residual_noise_offshore_highvar = self.rmse_residual_noise_offshore_highvar
            
            
        data = [['BEFORE FILTER', 
                 'SSH [m]',
                 self.rmse_karin_noise_global, 
                 self.rmse_karin_noise_coastal, 
                 self.rmse_karin_noise_offshore_lowvar, 
                 self.rmse_karin_noise_offshore_highvar,
                 np.round(wavelength_snr1_nofilter, 1),
                 notebook_name],
                ['BEFORE FILTER', 
                 'Geostrophic current [m.s$^-1$]',
                 self.rmse_karin_noise_global_ug, 
                 self.rmse_karin_noise_coastal_ug, 
                 self.rmse_karin_noise_offshore_lowvar_ug, 
                 self.rmse_karin_noise_offshore_highvar_ug,
                 np.round(self.wavelength_snr1_nofilter_ug, 1),
                 notebook_name],
                ['BEFORE FILTER', 
                 'Relative vorticity []',
                 self.rmse_karin_noise_global_ksi, 
                 self.rmse_karin_noise_coastal_ksi, 
                 self.rmse_karin_noise_offshore_lowvar_ksi, 
                 self.rmse_karin_noise_offshore_highvar_ksi,
                 np.round(self.wavelength_snr1_nofilter_ksi, 1),
                 notebook_name],
                [self.filter_name, 
                 'SSH [m]',
                 rmse_residual_noise_global, 
                 rmse_residual_noise_coastal, 
                 rmse_residual_noise_offshore_lowvar, 
                 rmse_residual_noise_offshore_highvar,
                 np.round(wavelength_snr1_filter, 1),
                 notebook_name], 
                [self.filter_name, 
                 'Geostrophic current [m.s$^-1$]',
                 self.rmse_residual_noise_global_ug, 
                 self.rmse_residual_noise_coastal_ug, 
                 self.rmse_residual_noise_offshore_lowvar_ug, 
                 self.rmse_residual_noise_offshore_highvar_ug,
                 np.round(self.wavelength_snr1_filter_ug, 1),
                 notebook_name],
                [self.filter_name, 
                 'Relative vorticity []',
                 self.rmse_residual_noise_global_ksi, 
                 self.rmse_residual_noise_coastal_ksi, 
                 self.rmse_residual_noise_offshore_lowvar_ksi, 
                 self.rmse_residual_noise_offshore_highvar_ksi,
                 np.round(self.wavelength_snr1_filter_ksi, 1),
                 notebook_name]
               
               ]
        
        Leaderboard = pd.DataFrame(data, 
                           columns=['Method',
                                    'Field',
                                    "µ(RMSE global)", 
                                    "µ(RMSE coastal)", 
                                    "µ(RMSE offshore lowvar)", 
                                    "µ(RMSE offshore highvar)", 
                                    'λ(SNR1) [km]', 
                                    'Reference'])
        print("Summary of the leaderboard metrics:")
        print(Leaderboard.to_markdown())
            
    
    

    

    @property
    def longitude(self):
        return self.stats_dict['ssh'].x

    @property
    def latitude(self):
        return self.stats_dict['ssh'].y
        
    @property
    def gridstep(self):
        return self._gridstep

    @property
    def stats_dict(self):
        return self._d_stats
    
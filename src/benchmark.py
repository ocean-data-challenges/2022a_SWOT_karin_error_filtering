import numpy as np
import xarray as xr
import os
import glob
import pyinterp
from scipy import signal
from itertools import chain
import hvplot.xarray
import pandas as pd
import warnings
warnings.filterwarnings("ignore")




class Benchmark(object):

    def __init__(self, gridstep=1):
        self._gridstep = gridstep
        self._stats = ['ssh','grad_ssh_across','grad_ssh_along','ssh_rmse','grad_ssh_across_rmse','grad_ssh_along_rmse']
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


    def compute_stats(self, l_files, refvar, etuvar):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: unfiltered (true) SSH variable name
            etuvar: filtered SSH variable name
        """

        for i, fname in enumerate(l_files):
            #swt = SwotTrack(fname)
            swt = xr.open_dataset(fname)

            # SSH
            self.stats_dict['ssh'].push(
                swt.longitude.values.flatten(),
                swt.latitude.values.flatten(),
                (swt[etuvar].values - swt[refvar].values).flatten(),
                False
            )
            
            # SSH RMSE
            self.stats_dict['ssh_rmse'].push(
                swt.longitude.values.flatten(),
                swt.latitude.values.flatten(),
                ((swt[etuvar].values - swt[refvar].values)**2).flatten(),
                False
            )
            
            f_coriolis = self._coriolis_parameter(swt.latitude.values)
            
            # gradients (approx geostrophic velocity, sign of the track is not checked)
            grad_ssh_along, grad_ssh_across = self._compute_grad_diff(
                swt[etuvar].values,
                swt[refvar].values,
                f_coriolis
            )
            self.stats_dict['grad_ssh_along'].push(
                swt.longitude.values.flatten(),
                swt.latitude.values.flatten(),
                grad_ssh_along.flatten(),
                False
            )
            self.stats_dict['grad_ssh_across'].push(
                swt.longitude.values.flatten(),
                swt.latitude.values.flatten(),
                grad_ssh_across.flatten(),
                False
            )
            
            self.stats_dict['grad_ssh_along_rmse'].push(
                swt.longitude.values.flatten(),
                swt.latitude.values.flatten(),
                (grad_ssh_along**2).flatten(),
                False
            )
            self.stats_dict['grad_ssh_across_rmse'].push(
                swt.longitude.values.flatten(),
                swt.latitude.values.flatten(),
                (grad_ssh_across**2).flatten(),
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
                ssh_mean=(["lat", "lon"], self.stats_dict['ssh'].variable('mean').T),
                ssh_variance=(["lat", "lon"], self.stats_dict['ssh'].variable('variance').T),
                ssh_count=(["lat", "lon"], self.stats_dict['ssh'].variable('count').T),
                ssh_minimum=(["lat", "lon"],self.stats_dict['ssh'].variable('min').T),
                ssh_maximum=(["lat", "lon"], self.stats_dict['ssh'].variable('max').T),
                ssh_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['ssh_rmse'].variable('mean').T)),
                grad_ssh_across_mean=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('mean').T),
                grad_ssh_across_variance=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('variance').T),
                grad_ssh_across_count=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('count').T),
                grad_ssh_across_minimum=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('min').T),
                grad_ssh_across_maximum=(["lat", "lon"], self.stats_dict['grad_ssh_across'].variable('max').T),
                grad_ssh_across_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['grad_ssh_across_rmse'].variable('mean').T)),
                grad_ssh_along_mean=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('mean').T),
                grad_ssh_along_variance=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('variance').T),
                grad_ssh_along_count=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('count').T),
                grad_ssh_along_minimum=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('min').T),
                grad_ssh_along_maximum=(["lat", "lon"], self.stats_dict['grad_ssh_along'].variable('max').T),
                grad_ssh_along_rmse=(["lat", "lon"], np.sqrt(self.stats_dict['grad_ssh_along_rmse'].variable('mean').T)),
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
        
        if 'filter' in kwargs:
            self.filter_name = kwargs['filter']
        else :
            self.filter_name = 'None'
        
    def display_stats(self, fname):
        
        
        ds = xr.open_dataset(fname)
        
        vmin = np.nanpercentile(ds['ssh_mean']*100., 5)
        vmax = np.nanpercentile(ds['ssh_mean']*100, 95)
        fig1 = (100*ds['ssh_mean']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='SSH residual MEAN').opts(frame_height=400, frame_width=400)

        vmin = np.nanpercentile(ds['ssh_variance']*10000., 5)
        vmax = np.nanpercentile(ds['ssh_variance']*10000, 95)
        fig2 = (10000*ds['ssh_variance']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='SSH residual VARIANCE').opts(frame_height=400, frame_width=400)
        
        vmin = np.nanpercentile(ds['ssh_rmse'], 5)
        vmax = np.nanpercentile(ds['ssh_rmse'], 95)
        fig3 = (ds['ssh_rmse']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='SSH residual RMSE').opts(frame_height=400, frame_width=400)

        vmin = np.nanpercentile(ds['grad_ssh_across_mean'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_across_mean'], 95)
        fig4 = (ds['grad_ssh_across_mean']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='grad_ac SSH residual MEAN').opts(frame_height=400, frame_width=400)

        vmin = np.nanpercentile(ds['grad_ssh_across_variance'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_across_variance'], 95)
        fig5 = (ds['grad_ssh_across_variance']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='grad_ac SSH residual VARIANCE').opts(frame_height=400, frame_width=400)
        
        vmin = np.nanpercentile(ds['grad_ssh_across_rmse'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_across_rmse'], 95)
        fig6 = (ds['grad_ssh_across_rmse']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='grad_ac SSH residual RMSE').opts(frame_height=400, frame_width=400)

        vmin = np.nanpercentile(ds['grad_ssh_along_mean'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_along_mean'], 95)
        fig7 = (ds['grad_ssh_along_mean']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='grad_al SSH residual MEAN').opts(frame_height=400, frame_width=400)

        vmin = np.nanpercentile(ds['grad_ssh_along_variance'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_along_variance'], 95)
        fig8 = (ds['grad_ssh_along_variance']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='grad_al SSH residual VARIANCE').opts(frame_height=400, frame_width=400)
        
        vmin = np.nanpercentile(ds['grad_ssh_along_rmse'], 5)
        vmax = np.nanpercentile(ds['grad_ssh_along_rmse'], 95)
        fig9 = (ds['grad_ssh_along_rmse']).hvplot.quadmesh(x='lon', y='lat', clim=(vmin, vmax), cmap='jet', geo=True, coastline=True, title='grad_al SSH residual RMSE').opts(frame_height=400, frame_width=400)

        return (fig1 + fig2 + fig3 + fig4 + fig5 + fig6 + fig7 + fig8 + fig9 ).cols(3)
    
    
#     def across_track_residual_noise(self, l_files, refvar, etuvar):
        
#         ds = xr.open_mfdataset(l_files, combine='nested', concat_dim='num_lines')
#         self.ac_residual_noise = np.sqrt((ds[etuvar] - ds[refvar]).var(dim='num_lines'))
#         self.ac_karin_noise = np.sqrt((ds['simulated_noise_ssh_karin'] - ds[refvar]).var(dim='num_lines'))
        
    
    def compute_stats_by_regime(self, l_files, refvar, etuvar):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: unfiltered (true) SSH variable name
            etuvar: filtered SSH variable name
        """
        ds = xr.open_mfdataset(l_files, combine='nested', concat_dim='num_lines')
        ds['residual_noise'] = ds[etuvar] - ds[refvar]
        ds['karin_noise'] = ds['simulated_noise_ssh_karin'] - ds[refvar]
        
        self.num_pixels = ds.num_pixels.values
        
        # GLOBAL STAT
        self.std_ac_residual_noise_global = np.sqrt((ds['residual_noise']).var(dim='num_lines')).values
        self.std_ac_karin_noise_global = np.sqrt((ds['karin_noise']).var(dim='num_lines')).values
        self.std_residual_noise_global = np.sqrt((ds['residual_noise']).var()).values
        self.std_karin_noise_global = np.sqrt((ds['karin_noise']).var()).values
        
        self.rmse_ac_residual_noise_global = np.sqrt( ((ds['residual_noise'])**2).mean(dim='num_lines') ).values
        self.rmse_residual_noise_global = np.sqrt( ((ds['residual_noise'])**2).mean() ).values
        
        # COASTAL (< 200 km) STAT 
        ds_tmp = ds.where(ds['distance_to_nearest_coastline'] <= 200., drop=True)
        self.std_ac_residual_noise_coastal = np.sqrt((ds_tmp['residual_noise']).var(dim='num_lines')).values
        self.std_ac_karin_noise_coastal = np.sqrt((ds_tmp['karin_noise']).var(dim='num_lines')).values
        self.std_residual_noise_coastal = np.sqrt((ds_tmp['residual_noise']).var()).values
        self.std_karin_noise_coastal = np.sqrt((ds_tmp['karin_noise']).var()).values
        
        self.rmse_ac_residual_noise_coastal = np.sqrt( ((ds_tmp['residual_noise'])**2).mean(dim='num_lines') ).values
        self.rmse_residual_noise_coastal = np.sqrt( ((ds_tmp['residual_noise'])**2).mean() ).values
        
        del ds_tmp
        
        # OFFSHORE (> 200 km) + LOW VARIABILITY ( < 200cm2) STAT 
        ds_tmp = ds.where((ds['distance_to_nearest_coastline'] >= 200.) & (ds['ssh_variance'] <= 0.0200), drop=True)
        self.std_ac_residual_noise_offshore_lowvar = np.sqrt((ds_tmp['residual_noise']).var(dim='num_lines')).values
        self.std_ac_karin_noise_offshore_lowvar = np.sqrt((ds_tmp['karin_noise']).var(dim='num_lines')).values
        self.std_residual_noise_offshore_lowvar = np.sqrt((ds_tmp['residual_noise']).var()).values
        self.std_karin_noise_offshore_lowvar = np.sqrt((ds_tmp['karin_noise']).var()).values
        
        self.rmse_ac_residual_noise_offshore_lowvar = np.sqrt( ((ds_tmp['residual_noise'])**2).mean(dim='num_lines') ).values
        self.rmse_residual_noise_offshore_lowvar = np.sqrt( ((ds_tmp['residual_noise'])**2).mean() ).values
        
        del ds_tmp
        
        # OFFSHORE (> 200 km) + LOW VARIABILITY ( < 200cm2) STAT 
        ds_tmp = ds.where((ds['distance_to_nearest_coastline'] >= 200.) & (ds['ssh_variance'] >= 0.0200) , drop=True)
        self.std_ac_residual_noise_offshore_highvar = np.sqrt((ds_tmp['residual_noise']).var(dim='num_lines')).values
        self.std_ac_karin_noise_offshore_highvar = np.sqrt((ds_tmp['karin_noise']).var(dim='num_lines')).values
        self.std_residual_noise_offshore_highvar = np.sqrt((ds_tmp['residual_noise']).var()).values
        self.std_karin_noise_offshore_highvar = np.sqrt((ds_tmp['karin_noise']).var()).values
        
        self.rmse_ac_residual_noise_offshore_highvar = np.sqrt( ((ds_tmp['residual_noise'])**2).mean(dim='num_lines') ).values
        self.rmse_residual_noise_offshore_highvar = np.sqrt( ((ds_tmp['residual_noise'])**2).mean() ).values
        
        del ds_tmp
        
        
    def write_stats_by_regime(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        
        to_write = xr.Dataset(
            data_vars=dict(
                std_ac_residual_noise_global=(["num_pixels"], self.std_ac_residual_noise_global),
                std_ac_karin_noise_global=(["num_pixels"], self.std_ac_karin_noise_global),
                std_residual_noise_global=(["x"], [self.std_residual_noise_global]),
                std_karin_noise_global=(["x"], [self.std_karin_noise_global]),
                rmse_ac_residual_noise_global=(["num_pixels"], self.rmse_ac_residual_noise_global),
                rmse_residual_noise_global=(["x"], [self.rmse_residual_noise_global]),
            
                std_ac_residual_noise_coastal=(["num_pixels"], self.std_ac_residual_noise_coastal),
                std_ac_karin_noise_coastal=(["num_pixels"], self.std_ac_karin_noise_coastal),
                std_residual_noise_coastal=(["x"], [self.std_residual_noise_coastal]),
                std_karin_noise_coastal=(["x"], [self.std_karin_noise_coastal]),
                rmse_ac_residual_noise_coastal=(["num_pixels"], self.rmse_ac_residual_noise_coastal),
                rmse_residual_noise_coastal=(["x"], [self.rmse_residual_noise_coastal]),
        
                std_ac_residual_noise_offshore_lowvar=(["num_pixels"], self.std_ac_residual_noise_offshore_lowvar),
                std_ac_karin_noise_offshore_lowvar=(["num_pixels"], self.std_ac_karin_noise_offshore_lowvar),
                std_residual_noise_offshore_lowvar=(["x"], [self.std_residual_noise_offshore_lowvar]),
                std_karin_noise_offshore_lowvar=(["x"], [self.std_karin_noise_offshore_lowvar]),
                rmse_ac_residual_noise_offshore_lowvar=(["num_pixels"], self.rmse_ac_residual_noise_offshore_lowvar),
                rmse_residual_noise_offshore_lowvar=(["x"], [self.rmse_residual_noise_offshore_lowvar]),
            
                std_ac_residual_noise_offshore_highvar=(["num_pixels"], self.std_ac_residual_noise_offshore_highvar),
                std_ac_karin_noise_offshore_highvar=(["num_pixels"], self.std_ac_karin_noise_offshore_highvar),
                std_residual_noise_offshore_highvar=(["x"], [self.std_residual_noise_offshore_highvar]),
                std_karin_noise_offshore_highvar=(["x"], [self.std_karin_noise_offshore_highvar]),
                rmse_ac_residual_noise_offshore_highvar=(["num_pixels"], self.rmse_ac_residual_noise_offshore_highvar),
                rmse_residual_noise_offshore_highvar=(["x"], [self.rmse_residual_noise_offshore_highvar]),
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
        
    
    def display_stats_by_regime(self, fname):
        
        ds = xr.open_dataset(fname).drop('x')
        
        fig1 = ds.std_ac_residual_noise_global.hvplot.line(x='num_pixels', ylim=(0, 0.025), label='residual_noise_global') * ds.std_ac_karin_noise_global.hvplot.line(x='num_pixels', label='karin_noise_global')
        fig2 = ds.std_ac_residual_noise_coastal.hvplot.line(x='num_pixels', ylim=(0, 0.025),label='residual_noise_coastal') * ds.std_ac_karin_noise_coastal.hvplot.line(x='num_pixels',label='karin_noise_coastal')
        fig3 = ds.std_ac_residual_noise_offshore_lowvar.hvplot.line(x='num_pixels', ylim=(0, 0.025), label='residual_noise_offshore_lowvar') * ds.std_ac_karin_noise_offshore_lowvar.hvplot.line(x='num_pixels', label='karin_noise_offshore_lowvar')
        fig4 = ds.std_ac_residual_noise_offshore_highvar.hvplot.line(x='num_pixels', ylim=(0, 0.025), label='residual_noise_offshore_highvar') * ds.std_ac_karin_noise_offshore_highvar.hvplot.line(x='num_pixels', label='karin_noise_offshore_highvar')
        
        return (fig1* fig2 *fig3 *fig4).opts(legend_position='right',frame_height=300, frame_width=500) 
            
    
    def compute_along_track_psd(self, l_files, refvar, etuvar, lengh_scale=512, overlay=0.25, details=False):
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
        resolution = 2. # along-track resolution in km
        npt = int(lengh_scale/resolution)
        n_overlap = int(npt*overlay)
        
        
        for i, fname in enumerate(l_files):
            #swt = SwotTrack(fname)
            swt = xr.open_dataset(fname)
        
            # parcours des différentes lignes along-track
            for ac_index in swt.num_pixels.values:
                # extraction des lon/lat/ssh
                lon = swt.longitude.values[:,ac_index]
                lat = swt.longitude.values[:,ac_index]
                ssh_true = swt[refvar].values[:,ac_index]
                ssh_noisy = swt.simulated_noise_ssh_karin.values[:,ac_index]
                ssh_filtered = swt[etuvar].values[:,ac_index]
            
                # construction de la liste des segments
                al_seg_ssh_true, al_seg_ssh_noisy, al_seg_ssh_filtered = create_segments_from_1d(lon, 
                                                                                                 lat, 
                                                                                                 ssh_true, 
                                                                                                 ssh_noisy, 
                                                                                                 ssh_filtered, 
                                                                                                 npt=npt,  
                                                                                                 n_overlay=n_overlap)
                l_segment_ssh_true.append(al_seg_ssh_true)
                l_segment_ssh_noisy.append(al_seg_ssh_noisy)
                l_segment_ssh_filtered.append(al_seg_ssh_filtered)
        
        # on met la liste à plat
        l_flat_ssh_true = np.asarray(list(chain.from_iterable(l_segment_ssh_true))).flatten()
        l_flat_ssh_noisy = np.asarray(list(chain.from_iterable(l_segment_ssh_noisy))).flatten()
        l_flat_ssh_filtered = np.asarray(list(chain.from_iterable(l_segment_ssh_filtered))).flatten()
        
        # PSD 
        freq, cross_spectrum = signal.csd(l_flat_ssh_noisy,l_flat_ssh_filtered, fs=1./resolution, nperseg=npt, noverlap=0)
        freq, psd_ssh_true  = signal.welch(l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ssh_noisy = signal.welch(l_flat_ssh_noisy, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_ssh_filtered = signal.welch(l_flat_ssh_filtered, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err = signal.welch(l_flat_ssh_filtered - l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        freq, psd_err_karin = signal.welch(l_flat_ssh_noisy - l_flat_ssh_true, fs=1./resolution, window='hann', nperseg=npt, noverlap=0, detrend='constant', return_onesided=True, scaling='density', average='mean')
        
        
        
        self.freq = freq
        self.cross_spectrum = cross_spectrum
        self.psd_ssh_true = psd_ssh_true
        self.psd_ssh_noisy = psd_ssh_noisy
        self.psd_ssh_filtered = psd_ssh_filtered
        self.psd_err = psd_err
        self.psd_err_karin = psd_err_karin
        
#         # on calcule les psd
#         l_psd = [segment.psd() for segment in l_flat]
        
#         # récupère le vecteur des fréquences (en cyc/km)
#         freq = np.array(l_psd[0][0])
        
#         # on met ça dans un numpy array
#         n_seg = len(l_flat)
#         n_freq = len(freq)
#         psd_array = np.empty((n_freq, n_seg), dtype=float)
        
#         for ii, (f, psd) in enumerate(l_psd):
#             psd_array[:,ii] = np.array(psd)

#         avg_psd = np.mean(psd_array, axis=-1)

        # if details:
        #     return freq, avg_psd, psd_array, l_flat
        # else:
        
        
        #return freq, psd_ssh_true, psd_ssh_noisy, psd_ssh_filtered, psd_err, psd_err_karin
    
    
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
        
            if len(zero_crossings) > 0:
                list_of_res = []
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
        
            resolution_scale = np.max(np.asarray(list_of_res))
        
            return resolution_scale#, flag_multiple_crossing
        
        self.wavelength_snr1_filter = compute_snr1(self.psd_err/self.psd_ssh_true, self.freq)
        self.wavelength_snr1_nofilter = compute_snr1(self.psd_err_karin/self.psd_ssh_true, self.freq)
        
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

            ),
            coords=dict(
                wavenumber=(["wavenumber"], self.freq),
                wavelength_snr1_filter=(["wavelength_snr1"], [self.wavelength_snr1_filter]),
                wavelength_snr1_nofilter=(["wavelength_snr1"], [self.wavelength_snr1_nofilter]),
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
                
        data = [[self.filter_name, 
                 rmse_residual_noise_global, 
                 rmse_residual_noise_coastal, 
                 rmse_residual_noise_offshore_lowvar, 
                 rmse_residual_noise_offshore_highvar,
                 np.round(wavelength_snr1_nofilter, 1),
                 np.round(wavelength_snr1_filter, 1),
                 notebook_name]]
        
        Leaderboard = pd.DataFrame(data, 
                           columns=['Method', 
                                    "µ(RMSE global) [m]", 
                                    "µ(RMSE coastal) [m]", 
                                    "µ(RMSE offshore lowvar) [m]", 
                                    "µ(RMSE offshore highvar) [m]", 
                                    'λ(SNR1 before filtering) [km]', 
                                    'λ(SNR1 after filtering) [km]',  
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
    
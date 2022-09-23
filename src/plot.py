import numpy as np
import hvplot.xarray
import xarray as xr
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .swot import *


def display_track(ds):
    vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
    vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
    fig_ssh_true = ds['simulated_true_ssh_karin'].hvplot.quadmesh(x='longitude', y='latitude', clim=(vmin, vmax), cmap='Spectral_r', rasterize=True, title='True ssh karin')
    fig_noisy_ssh = ds['simulated_noise_ssh_karin'].hvplot.quadmesh(x='longitude', y='latitude', clim=(vmin, vmax), cmap='Spectral_r', rasterize=True, title='Noisy ssh karin')
    fig_filtered_ssh = ds['ssh_karin_filt'].hvplot.quadmesh(x='longitude', y='latitude', clim=(vmin, vmax), cmap='Spectral_r', rasterize=True, title='Filtered ssh karin')
    
    delta = ds['ssh_karin_filt'] - ds['simulated_true_ssh_karin']
    vmin_delta = np.nanpercentile(delta.values, 5)
    vmax_delta = np.nanpercentile(delta.values, 95)
    fig_delta_ssh_filtered_ssh_true = delta.hvplot.quadmesh(x='longitude', y='latitude', clim=(-np.abs(vmin_delta), np.abs(vmin_delta)), cmap='bwr', rasterize=True, title='Filtered ssh karin - True ssh karin')
    
    return (fig_ssh_true + fig_noisy_ssh + fig_filtered_ssh + fig_delta_ssh_filtered_ssh_true).cols(2)



def compare_stat(filename_ref, filename_etu, **kwargs):
    
    ds_ref = xr.open_dataset(filename_ref)
    ref_filter = ds_ref.filter_type
    ds_etu = xr.open_dataset(filename_etu)
    etu_filter = ds_etu.filter_type
    
    ds = 100*(ds_etu - ds_ref)/ds_ref
    
    plt.figure(figsize=(18, 15))

        
    ax = plt.subplot(311, projection=ccrs.PlateCarree())
    vmin = np.nanpercentile(ds.ssh_rmse, 5)
    vmax = np.nanpercentile(ds.ssh_rmse, 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    ds.ssh_rmse.plot(x='lon', y='lat', vmin=-vmin, vmax=vmin, cmap='bwr', cbar_kwargs={'label': '[%]'}, **kwargs)
    plt.title('$\Delta$ RMSE SSH field ' + f'{etu_filter} vs {ref_filter}', fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=2)
    ax.coastlines(zorder=2)

    ax = plt.subplot(312, projection=ccrs.PlateCarree())
    vmin = np.nanpercentile(ds.ug_rmse, 5)
    vmax = np.nanpercentile(ds.ug_rmse, 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    ds.ug_rmse.plot(x='lon', y='lat', vmin=-vmin, vmax=vmin, cmap='bwr', cbar_kwargs={'label': '[%]'}, **kwargs)
    plt.title('$\Delta$ RMSE GEOSTROPPHIC CURRENT field ' + f'{etu_filter} vs {ref_filter}', fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=2)
    ax.coastlines(zorder=2)
        
    ax = plt.subplot(313, projection=ccrs.PlateCarree())
    vmin = np.nanpercentile(ds.ksi_rmse, 5)
    vmax = np.nanpercentile(ds.ksi_rmse, 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    ds.ksi_rmse.plot(x='lon', y='lat', vmin=-vmin, vmax=vmin, cmap='bwr', cbar_kwargs={'label': '[%]'}, **kwargs)
    plt.title('$\Delta$ RMSE Relative vorticity '+ f'{etu_filter} vs {ref_filter}', fontweight='bold')
    ax.add_feature(cfeature.LAND, zorder=2)
    ax.coastlines(zorder=2)

    plt.show()
    
def compare_stats_by_regime(list_of_filename, list_of_label): 
    
    ds = xr.concat([xr.open_dataset(filename) for filename in list_of_filename], dim='experiment')
    ds['experiment'] = list_of_label

    fig = plt.figure(figsize=(12, 13))
    
    plt.subplot(221)
    
    plt.plot(2*ds.num_pixels - 70, ds['std_ac_karin_noise_global'][0, :].values, c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['std_ac_residual_noise_global'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('GLOBAL', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    
    plt.subplot(222)
    plt.plot(2*ds.num_pixels - 70, ds['std_ac_karin_noise_coastal'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['std_ac_residual_noise_coastal'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('COASTAL', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    
    plt.subplot(223)
    plt.plot(2*ds.num_pixels - 70, ds['std_ac_karin_noise_offshore_lowvar'][0, :].values,c='k', lw=2, label='karin_noise_offshore_lowvar')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['std_ac_residual_noise_offshore_lowvar'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n LOW VARIBILITY (< 200cm$^2$)', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    
    plt.subplot(224)
    plt.plot(2*ds.num_pixels - 70, ds['std_ac_karin_noise_offshore_highvar'][0, :].values,c='k', lw=2, label='karin_noise')
    for exp in ds['experiment'].values:
        ds_sel = ds.where(ds.experiment == exp, drop=True)
        plt.plot(2*ds_sel.num_pixels - 70, ds_sel['std_ac_residual_noise_offshore_highvar'].squeeze(), label=f'residual_noise ({exp})')
    plt.title('OFFSHORE (> 200km),\n HIGH VARIBILITY (> 200cm$^2$)', fontweight='bold')
    plt.ylabel('Height Error [m]')  
    plt.xlabel('Ground Range [km]')
    plt.legend(loc='best')
    plt.grid()
    plt.ylim(0, 0.03)
    #plt.axis([0,72,0,0.03])    

    
    



def compare_psd(list_of_filename, list_of_label):
        
    ds = xr.concat([xr.open_dataset(filename) for filename in list_of_filename], dim='experiment')
    ds['experiment'] = list_of_label
    ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})

    fig = plt.figure(figsize=(15, 18))

    ax = plt.subplot(321)
    ds['psd_ssh_true'][0, :].plot(x='wavelength', label='PSD(SSH$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
    ds['psd_ssh_noisy'][0, :].plot(x='wavelength', label='PSD(SSH$_{noisy}$)', color='r', lw=2)
    for exp in ds['experiment'].values:
        (ds['psd_ssh_filtered'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(SSH$_{filtered}$)' + f'({exp})', lw=2)
        (ds['psd_err'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(SSH$_{err}$)' + f'({exp})', lw=2)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('PSD [m.cy$^{-1}$.km$^{-1}$]')
    ax.invert_xaxis()
    plt.title('PSD Sea Surface Height')

    ds['SNR_filter'] = ds['psd_err']/ds['psd_ssh_true']
    ds['SNR_nofilter'] = ds['psd_err_karin']/ds['psd_ssh_true']
    ax = plt.subplot(322)
    for exp in ds['experiment'].values:
        (ds['SNR_filter'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(SSH$_{err}$)/PSD(SSH$_{true}$)' + f'({exp})', xscale='log', lw=3)
        plt.scatter(ds.wavelength_snr1_filter.where(ds['experiment']==exp, drop=True), 1., zorder=4, label="SNR1 AFTER filter" + f'({exp})')
    ds['SNR_nofilter'][0, :].plot(x='wavelength', label='PSD(Karin$_{noise}$)/PSD(SSH$_{true}$)', color='r', lw=2)
    (ds['SNR_filter'][0, :]/ds['SNR_filter'][0, :]).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
    plt.scatter(ds.wavelength_snr1_nofilter[0, :], 1., color='r', zorder=4, label="SNR1 BEFORE filter")
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('SNR')

    plt.ylim(0, 2)
    ax.invert_xaxis()
    plt.title('SNR Sea Surface Height')


    ax = plt.subplot(323)
    ds['psd_ug_true'][0, :].plot(x='wavelength', label='PSD(Ug$_{true}$)', color='k', xscale='log', yscale='log', lw=3)
    ds['psd_ug_noisy'][0, :].plot(x='wavelength', label='PSD(Ug$_{noisy}$)', color='r', lw=2)
    for exp in ds['experiment'].values:
        (ds['psd_ug_filtered'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(Ug$_{filtered}$)' + f'({exp})', lw=2)
        (ds['psd_err_ug'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label=f'PSD(err)' + f'({exp})', lw=2)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('PSD [m.s$^{-1}$.cy$^{-1}$.km$^{-1}$]')
    ax.invert_xaxis()
    plt.title('PSD Geostrophic current')

    ds['SNR_filter_ug'] = ds['psd_err_ug']/ds['psd_ug_true']
    ds['SNR_nofilter_ug'] = ds['psd_err_karin_ug']/ds['psd_ug_true']
    ax = plt.subplot(324)
    for exp in ds['experiment'].values:
        (ds['SNR_filter_ug'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD(Ug$_{err}$)/PSD(Ug$_{true}$)' + f'({exp})', xscale='log', lw=3)
        plt.scatter(ds.wavelength_snr1_filter_ug.where(ds['experiment']==exp, drop=True), 1., zorder=4, label="SNR1 AFTER filter" + f'({exp})')
    ds['SNR_nofilter_ug'][0, :].plot(x='wavelength', label='PSD(Ug$_{noise}$)/PSD(Ug$_{true}$)', color='r', lw=2)
    (ds['SNR_filter_ug'][0, :]/ds['SNR_filter_ug'][0, :]).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
    plt.scatter(ds.wavelength_snr1_nofilter_ug[0, :], 1., color='r', zorder=4, label="SNR1 BEFORE filter")
    plt.grid(which='both')
    plt.legend()
    plt.ylim(0, 2)
    ax.invert_xaxis()
    plt.title('SNR Geostrophic current')
    plt.xlabel('wavelenght [km]')
    plt.ylabel('SNR')


    ax = plt.subplot(325)
    ds['psd_ksi_true'][0, :].plot(x='wavelength', label='PSD($\zeta_{true}$)', color='k', xscale='log', yscale='log', lw=3)
    ds['psd_ksi_noisy'][0, :].plot(x='wavelength', label='PSD($\zeta_{noisy}$)', color='r', lw=2)
    for exp in ds['experiment'].values:
        (ds['psd_ksi_filtered'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD($\zeta_{filtered}$)' + f'({exp})', lw=2)
        (ds['psd_err_ksi'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label=f'PSD(err)' + f'({exp})', lw=2)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('wavelenght [km]')
    plt.ylabel('PSD [s$^{-1}$.cy$^{-1}$.km$^{-1}$]')
    ax.invert_xaxis()
    plt.title('PSD Relative vorticity')

    ds['SNR_filter_ksi'] = ds['psd_err_ksi']/ds['psd_ksi_true']
    ds['SNR_nofilter_ksi'] = ds['psd_err_karin_ksi']/ds['psd_ksi_true']
    ax = plt.subplot(326)
    for exp in ds['experiment'].values:
        (ds['SNR_filter_ksi'].where(ds['experiment']==exp, drop=True)).plot(x='wavelength', label='PSD($\zeta_{err}$)/PSD($\zeta_{true}$)' + f'({exp})', xscale='log', lw=3)
        plt.scatter(ds.wavelength_snr1_filter_ksi.where(ds['experiment']==exp, drop=True), 1., zorder=4, label="SNR1 AFTER filter" + f'({exp})')
    ds['SNR_nofilter_ksi'][0, :].plot(x='wavelength', label='PSD($\zeta_{noise}$)/PSD($\zeta_{true}$)', color='r', lw=2)
    (ds['SNR_filter_ksi'][0, :]/ds['SNR_filter_ksi'][0, :]).plot(x='wavelength', label='SNR=1', color='grey', lw=2)
    
    plt.scatter(ds.wavelength_snr1_nofilter_ksi[0, :], 1., color='r', zorder=4, label="SNR1 BEFORE filter")
    plt.grid(which='both')
    plt.legend()
    plt.ylim(0, 2)
    ax.invert_xaxis()
    plt.title('SNR Relative vorticity')
    plt.xlabel('wavelenght [km]')
    plt.ylabel('SNR')
        
    plt.show()
    
    
    
def plot_demo_pass(file_ref_input, file_filtered):
    swt_input = SwotTrack(file_ref_input)
    swt_input.compute_geos_current('ssh_true', 'true_geos_current')
    swt_input.compute_relative_vorticity('true_geos_current_x', 'true_geos_current_y', 'true_ksi')
    swt_input.compute_geos_current('ssh_karin', 'karin_geos_current')
    swt_input.compute_relative_vorticity('karin_geos_current_x', 'karin_geos_current_y', 'karin_ksi')
    
    
    swt_filtered = SwotTrack(file_filtered)
    swt_filtered.compute_geos_current('ssh_karin_filt', 'filtered_geos_current')
    swt_filtered.compute_relative_vorticity('filtered_geos_current_x', 'filtered_geos_current_y', 'filtered_ksi')
    
    
    n_p, n_l = np.meshgrid(swt_input._dset.num_pixels.values,swt_input._dset.num_lines.values)
    
    fig, axs = plt.subplots(2, 3,figsize=(10,16))
    row,col = 0,0
    ax = axs[row,col]
    ssh_true = swt_input._dset.ssh_true.values
    msk = np.isnan(swt_input._dset.ssh_karin)
    ssh_true[msk] = np.nan
    vmin = np.nanpercentile(ssh_true, 5)
    vmax = np.nanpercentile(ssh_true, 95)
    ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True SSH')
    ax.set_ylim(2000, 3000)

    row,col = 1,0
    ax = axs[row,col] 
    pcm =ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True SSH')
    ax.set_ylim(2000, 3000)

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6,location='left', label='[m]') 


    row,col = 0,1
    ax = axs[row,col]
    ssh_karin = swt_input._dset.ssh_karin.values
    ax.pcolormesh(n_p,n_l,ssh_karin,vmin=vmin, vmax=vmax, cmap='Spectral_r') 
    ax.title.set_text('Noisy karin SSH')
    ax.set_ylim(2000, 3000)

    row,col = 1,1
    ax = axs[row,col]
    ssh_karin_filtered = swt_filtered._dset.ssh_karin_filt.values
    pcm =ax.pcolormesh(n_p,n_l, ssh_karin_filtered, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('Filtered SSH')
    ax.set_ylim(2000, 3000)

    cb=fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m]') 
    cb.remove()

    row,col = 0,2
    ax = axs[row,col]
    ssh_diff = ssh_true - ssh_karin
    vmin = np.nanpercentile(ssh_diff, 5)
    vmax = np.nanpercentile(ssh_diff, 95)
    vdata = np.maximum(np.abs(vmin), np.abs(vmax))
    ax.pcolormesh(n_p,n_l,ssh_diff,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Karin noise')
    ax.set_ylim(2000, 3000)

    row,col = 1,2
    ax = axs[row,col]
    ssh_diff_filtered = ssh_true-ssh_karin_filtered
    pcm =ax.pcolormesh(n_p,n_l,ssh_diff_filtered,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Residual noise')
    ax.set_ylim(2000, 3000)

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m]') 


    plt.subplots_adjust(left=0.25,wspace=0.5,right=0.8) 

    fig.show()
    
    
    
    
    
    fig, axs = plt.subplots(2, 3,figsize=(10,16))
    row,col = 0,0
    ax = axs[row,col]
    ssh_true = swt_input._dset.true_geos_current.values
    msk = np.isnan(swt_input._dset.ssh_karin)
    ssh_true[msk] = np.nan
    vmin = np.nanpercentile(ssh_true, 5)
    vmax = np.nanpercentile(ssh_true, 95)
    ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True Ug')
    ax.set_ylim(2000, 3000)

    row,col = 1,0
    ax = axs[row,col] 
    pcm =ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True Ug')
    ax.set_ylim(2000, 3000)

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6,location='left', label='[m.s$^{-1}$]') 


    row,col = 0,1
    ax = axs[row,col]
    ssh_karin = swt_input._dset.karin_geos_current.values
    ax.pcolormesh(n_p,n_l,ssh_karin,vmin=vmin, vmax=vmax, cmap='Spectral_r') 
    ax.title.set_text('Noisy karin Ug')
    ax.set_ylim(2000, 3000)

    row,col = 1,1
    ax = axs[row,col]
    ssh_karin_filtered = swt_filtered._dset.filtered_geos_current.values
    pcm =ax.pcolormesh(n_p,n_l, ssh_karin_filtered, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('Filtered Ug')
    ax.set_ylim(2000, 3000)

    cb=fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m.s$^{-1}$]') 
    cb.remove()

    row,col = 0,2
    ax = axs[row,col]
    ssh_diff = ssh_true - ssh_karin
    vmin = np.nanpercentile(ssh_diff, 5)
    vmax = np.nanpercentile(ssh_diff, 95)
    vdata = np.maximum(np.abs(vmin), np.abs(vmax))
    ax.pcolormesh(n_p,n_l,ssh_diff,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Karin noise Ug')
    ax.set_ylim(2000, 3000)

    row,col = 1,2
    ax = axs[row,col]
    ssh_diff_filtered = ssh_true-ssh_karin_filtered
    pcm =ax.pcolormesh(n_p,n_l,ssh_diff_filtered,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Residual noise Ug')
    ax.set_ylim(2000, 3000)
    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[m.s$^{-1}$]') 


    plt.subplots_adjust(left=0.25,wspace=0.5,right=0.8) 

    fig.show()
    
    
    
    
    
    
    
    
    
    
    fig, axs = plt.subplots(2, 3,figsize=(10,16))
    row,col = 0,0
    ax = axs[row,col]
    ssh_true = swt_input._dset.true_ksi.values
    msk = np.isnan(swt_input._dset.ssh_karin)
    ssh_true[msk] = np.nan
    vmin = np.nanpercentile(ssh_true, 5)
    vmax = np.nanpercentile(ssh_true, 95)
    ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True vorticity')
    ax.set_ylim(2000, 3000)

    row,col = 1,0
    ax = axs[row,col] 
    pcm =ax.pcolormesh(n_p,n_l,ssh_true, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('True vorticity')
    ax.set_ylim(2000, 3000)

    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6,location='left', label='[]') 


    row,col = 0,1
    ax = axs[row,col]
    ssh_karin = swt_input._dset.karin_ksi.values
    ax.pcolormesh(n_p,n_l,ssh_karin,vmin=vmin, vmax=vmax, cmap='Spectral_r') 
    ax.title.set_text('Noisy karin vorticity')
    ax.set_ylim(2000, 3000)

    row,col = 1,1
    ax = axs[row,col]
    ssh_karin_filtered = swt_filtered._dset.filtered_ksi.values
    pcm =ax.pcolormesh(n_p,n_l, ssh_karin_filtered, vmin=vmin, vmax=vmax,cmap='Spectral_r') 
    ax.title.set_text('Filtered vorticity')
    ax.set_ylim(2000, 3000)
    
    cb=fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[]') 
    cb.remove()

    row,col = 0,2
    ax = axs[row,col]
    ssh_diff = ssh_true - ssh_karin
    vmin = np.nanpercentile(ssh_diff, 5)
    vmax = np.nanpercentile(ssh_diff, 95)
    vdata = np.maximum(np.abs(vmin), np.abs(vmax))
    ax.pcolormesh(n_p,n_l,ssh_diff,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Karin noise vorticity')
    ax.set_ylim(2000, 3000)
    
    row,col = 1,2
    ax = axs[row,col]
    ssh_diff_filtered = ssh_true-ssh_karin_filtered
    pcm =ax.pcolormesh(n_p,n_l,ssh_diff_filtered,vmin=-vdata,vmax=vdata,cmap='bwr') 
    ax.title.set_text('Residual noise vorticity')
    ax.set_ylim(2000, 3000)
    fig.colorbar(pcm, ax=axs[:,col], shrink=0.6, label='[]') 


    plt.subplots_adjust(left=0.25,wspace=0.5,right=0.8) 
    
    fig.show()

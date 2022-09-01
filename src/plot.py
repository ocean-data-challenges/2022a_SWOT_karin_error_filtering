import numpy as np
import hvplot.xarray
import xarray as xr


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



def compare_stat(filename_ref, filename_etu):
    
    ds_ref = xr.open_dataset(filename_ref)
    ds_etu = xr.open_dataset(filename_etu)
    
    ds = 100*(ds_etu - ds_ref)/ds_ref
        
    vmin = np.nanpercentile(ds['ssh_mean'], 5)
    vmax = np.nanpercentile(ds['ssh_mean'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig1 = (ds['ssh_mean']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='ΔSSH residual MEAN [%]').opts(frame_height=400, frame_width=400)

    vmin = np.nanpercentile(ds['ssh_variance'], 5)
    vmax = np.nanpercentile(ds['ssh_variance'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig2 = (ds['ssh_variance']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='ΔSSH residual VARIANCE [%]').opts(frame_height=400, frame_width=400)
    
    vmin = np.nanpercentile(ds['ssh_rmse'], 5)
    vmax = np.nanpercentile(ds['ssh_rmse'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig3 = (ds['ssh_rmse']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='ΔSSH residual RMSE [%]').opts(frame_height=400, frame_width=400)

    vmin = np.nanpercentile(ds['grad_ssh_across_mean'], 5)
    vmax = np.nanpercentile(ds['grad_ssh_across_mean'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig4 = (ds['grad_ssh_across_mean']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='Δgrad_ac SSH residual MEAN [%]').opts(frame_height=400, frame_width=400)

    vmin = np.nanpercentile(ds['grad_ssh_across_variance'], 5)
    vmax = np.nanpercentile(ds['grad_ssh_across_variance'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig5 = (ds['grad_ssh_across_variance']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='Δgrad_ac SSH residual VARIANCE [%]').opts(frame_height=400, frame_width=400)
    
    vmin = np.nanpercentile(ds['grad_ssh_across_rmse'], 5)
    vmax = np.nanpercentile(ds['grad_ssh_across_rmse'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig6 = (ds['grad_ssh_across_rmse']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='Δgrad_ac SSH residual RMSE [%]').opts(frame_height=400, frame_width=400)

    vmin = np.nanpercentile(ds['grad_ssh_along_mean'], 5)
    vmax = np.nanpercentile(ds['grad_ssh_along_mean'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig7 = (ds['grad_ssh_along_mean']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='Δgrad_al SSH residual MEAN [%]').opts(frame_height=400, frame_width=400)

    vmin = np.nanpercentile(ds['grad_ssh_along_variance'], 5)
    vmax = np.nanpercentile(ds['grad_ssh_along_variance'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig8 = (ds['grad_ssh_along_variance']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='Δgrad_al SSH residual VARIANCE [%]').opts(frame_height=400, frame_width=400)
    
    vmin = np.nanpercentile(ds['grad_ssh_along_rmse'], 5)
    vmax = np.nanpercentile(ds['grad_ssh_along_rmse'], 95)
    vmin = np.maximum(np.abs(vmin), np.abs(vmax))
    fig9 = (ds['grad_ssh_along_rmse']).hvplot.quadmesh(x='lon', y='lat', clim=(-vmin, vmin), cmap='coolwarm', geo=True, coastline=True, title='Δgrad_al SSH residual RMSE [%]').opts(frame_height=400, frame_width=400)

    return (fig1 + fig2 + fig3 + fig4 + fig5 + fig6 + fig7 + fig8 + fig9).cols(3)
    
    
def compare_psd(list_of_filename, list_of_label):
    
    ds = xr.concat([xr.open_dataset(filename) for filename in list_of_filename], dim='experiment')
    ds['experiment'] = list_of_label
    
    ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})
        
    fig1 = ds['psd_ssh_true'].hvplot.line(x='wavelength', label='psd_ssh_true', loglog=True, flip_xaxis=True, grid=True, legend=True, line_width=4, line_color='k')*\
    ds['psd_ssh_noisy'][0, :].hvplot.line(x='wavelength', label='psd_ssh_noisy', line_color='r', line_width=3)*\
    ds['psd_ssh_filtered'].hvplot.line(x='wavelength', label='psd_ssh_filtered', line_width=3, by='experiment')*\
    ds['psd_err'][0, :].hvplot.line(x='wavelength', label='psd_err', line_color='grey', line_width=3).opts(title='PSD', frame_height=500, frame_width=400, xlabel='wavelength [km]', ylabel='PSD [m2/cy/km]')
        
    ds['Transfer_function'] = np.sqrt(ds.cross_spectrum_r**2 + ds.cross_spectrum_i**2)/ds.psd_ssh_noisy
    fig2 = (ds['Transfer_function'].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, ylim=(0., 1), grid=True, label='Transfer function', legend=True, line_width=3, by='experiment')*\
                (0.5*ds['Transfer_function']/ds['Transfer_function'])[0, :].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, label='Tf=0.5', legend=True, line_width=1, line_color='r')).opts(title='Transfer function', frame_height=500, frame_width=400, xlabel='wavelength [km]', ylabel='Transfer function')
        
    ds['SNR_filter'] = ds['psd_err']/ds['psd_ssh_true']
    ds['SNR_nofilter'] = ds['psd_err_karin']/ds['psd_ssh_true']
    
    fig3 = (
        ds['SNR_filter'].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, ylim=(0., 2), grid=True, label='SNR filter', legend=True, line_width=3, by='experiment')*\
        ds['SNR_nofilter'].hvplot.line(x='wavelength', logx=True, flip_xaxis=True, grid=True, label='SNR nofilter', legend=True, line_width=3)*\
        ds.snr1_filter.hvplot.scatter(x='wavelength_snr1_filter', color='k', by='experiment')*\
        ds.snr1_filter.hvplot.scatter(x='wavelength_snr1_nofilter', color='k', by='experiment')*\
        (ds['SNR_filter']/ds['SNR_filter']).hvplot.line(x='wavelength', logx=True, flip_xaxis=True, label='SNR=1', legend=True, line_width=1, line_color='k')
    ).opts(title='SNR', frame_height=500, frame_width=400, xlabel='wavelength [km]', ylabel='SNR')
    
    
    return (fig1 + fig2+fig3).cols(3)
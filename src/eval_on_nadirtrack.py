import xarray as xr
import numpy as np
import sys
import os
import pyinterp
import pyinterp.fill
import matplotlib.pylab as plt
from scipy import interpolate
sys.path.append('..')  




class Benchmark_onNadirs(object):

    def __init__(self, gridstep=1):
        self._gridstep = gridstep
        self._stats = ['lon','lat','x_acalong','ssh_true_rmse', 'ssh_err_rmse', 'ssh_calib_rmse']
        self._d_stats = dict()
        
    def _init_accumulators(self,ds):
        """ creation des accumulateurs """ 

        for k in self._stats:
            self._d_stats[k] = np.zeros_like(ds.time.values)


    def compute_stats(self, l_files, etuvar, l_files_input):
        """ 
        calcul des stats 
        INPUT:
            l_files: list of swot track files
            refavr: uncalibrated (true) SSH variable name
            etuvar: calibrated SSH variable name
        """

        self.mean_ssh_rmse = 0 
        self.mean_ssh_nocalib_rmse = 0
        
        ds = xr.open_dataset(l_files)
        
        ds_input = xr.open_dataset(l_files_input)
        
        self._init_accumulators(ds)
        
        self.stats_dict['lon']= ds.lonalong.values
        self.stats_dict['lat'] = ds.latalong.values
        self.stats_dict['x_acalong'] = ds.x_acalong.values
        
        # SSH RMSE
        self.stats_dict['ssh_true_rmse']= np.abs(ds_input.sshalong_true.values - ds_input.refalong.values) 
        self.stats_dict['ssh_err_rmse'] = np.abs(ds_input.sshalong_err.values - ds_input.refalong.values) 
        self.stats_dict['ssh_calib_rmse']=np.abs(ds[etuvar].values - ds_input.refalong.values) 
        
        ssh_true_rmse = ((ds_input.sshalong_true.values - ds_input.refalong.values)**2).flatten()
        ssh_true_rmse = ssh_true_rmse[~np.isnan(ssh_true_rmse)]
        self.mean_ssh_true_rmse = np.sqrt(np.mean(ssh_true_rmse))
        

        ssh_nocalib_rmse = ((ds_input.sshalong_err.values - ds_input.refalong.values)**2).flatten()
        ssh_nocalib_rmse = ssh_nocalib_rmse[~np.isnan(ssh_nocalib_rmse)]
        self.mean_ssh_nocalib_rmse = np.sqrt(np.mean(ssh_nocalib_rmse))
            
        

        ssh_calib_rmse = ((ds[etuvar].values - ds_input.refalong.values)**2).flatten()
        ssh_calib_rmse = ssh_calib_rmse[~np.isnan(ssh_calib_rmse)]
        self.mean_ssh_calib_rmse = np.sqrt(np.mean(ssh_calib_rmse))
            
        
        self.stats_dict['mean_ssh_true_rmse']= self.mean_ssh_true_rmse
        self.stats_dict['mean_ssh_nocalib_rmse'] = self.mean_ssh_nocalib_rmse
        self.stats_dict['mean_ssh_calib_rmse']=self.mean_ssh_calib_rmse
        
             
    def write_stats(self, fname, **kwargs):
        """ export results on NetCDF fie """
        
        to_write = xr.Dataset(
            data_vars=dict(
                lon=(["time"], self.stats_dict['lon']),
                lat=(["time"], self.stats_dict['lat']),
                x_acalong=(["time"], self.stats_dict['x_acalong']),
                ssh_true_rmse=(["time"], self.stats_dict['ssh_true_rmse']),
                ssh_err_rmse=(["time"], self.stats_dict['ssh_err_rmse']),
                ssh_calib_rmse=(["time"], self.stats_dict['ssh_calib_rmse']),
                mean_ssh_true_rmse= self.mean_ssh_true_rmse,
                mean_ssh_err_rmse= self.mean_ssh_nocalib_rmse,
                mean_ssh_calib_rmse=self.mean_ssh_calib_rmse,
            ),
            coords=dict(
                time=(["time"], np.arange(np.shape(self.stats_dict['ssh_true_rmse'])[0])) 
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'statistics_of_residuals',
                calib_type=kwargs['calib'] if 'calib' in kwargs else 'None',
            ),
        )
        to_write.to_netcdf(fname) 
        
        if 'calib' in kwargs:
            self.calib_name = kwargs['calib']
        else :
            self.calib_name = 'None'
            
        
        
    def display_stats(self, fname, calib, **kwargs):
        
        ds = xr.open_dataset(fname)
        plt.figure(figsize=(5,4)) 
        plt.scatter(ds.lon-360,ds.lat,c=ds.ssh_true_rmse,vmin=0,vmax=0.1,cmap='Reds')
        plt.title('Residu SWOT true',fontsize=17)
        plt.xticks(plt.xticks()[0],np.array(np.abs(plt.xticks()[0]),dtype=int),fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('°W',fontsize=15)
        plt.ylabel('°N',fontsize=15)
        cbar = plt.colorbar() 
        cbar.ax.tick_params(labelsize=14)

        plt.figure(figsize=(5,4)) 
        plt.scatter(ds.lon-360,ds.lat,c=ds.ssh_err_rmse,vmin=0,vmax=0.1,cmap='Reds')
        plt.title('Residu SWOT error',fontsize=17)
        plt.xticks(plt.xticks()[0],np.array(np.abs(plt.xticks()[0]),dtype=int),fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('°W',fontsize=15)
        plt.ylabel('°N',fontsize=15)
        cbar = plt.colorbar() 
        cbar.ax.tick_params(labelsize=14)

        plt.figure(figsize=(5,4)) 
        plt.scatter(ds.lon-360,ds.lat,c=ds.ssh_calib_rmse,vmin=0,vmax=0.1,cmap='Reds')
        plt.title('Residu SWOT '+calib,fontsize=17)
        plt.xticks(plt.xticks()[0],np.array(np.abs(plt.xticks()[0]),dtype=int),fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('°W',fontsize=15)
        plt.ylabel('°N',fontsize=15)
        cbar = plt.colorbar() 
        cbar.ax.tick_params(labelsize=14)

        plt.show()
    
        print('---------------------------------') 
        print(color.BOLD+color.RED+'Global RMSE'+color.END)
        print('RMSE True SWOT:',ds.mean_ssh_true_rmse.values)
        print('RMSE Err SWOT:',ds.mean_ssh_err_rmse.values)
        print('RMSE '+calib+' SWOT:',ds.mean_ssh_calib_rmse.values)
        
        
    def display_stats_acrosstrack(self, fname, calib, **kwargs):
        
        nac = 120

        ds = xr.open_dataset(fname)
        
        rmse_true_ac = np.zeros(nac)
        rmse_err_ac = np.zeros(nac)
        rmse_cal_ac = np.zeros(nac)
        norm_ac = np.zeros(nac)

        for i in range(np.shape(ds.ssh_true_rmse)[0]):
            if ~np.isnan(np.round(ds.x_acalong[i]) ) and ~np.isnan(ds.ssh_true_rmse[i]):
                rmse_true_ac[int(np.round(ds.x_acalong[i]))] += ds.ssh_true_rmse[i]**2
                rmse_err_ac[int(np.round(ds.x_acalong[i]))] += ds.ssh_err_rmse[i]**2
                rmse_cal_ac[int(np.round(ds.x_acalong[i]))] += ds.ssh_calib_rmse[i]**2
                norm_ac[int(np.round(ds.x_acalong[i]))] += 1

        rmse_true_ac = np.sqrt(rmse_true_ac/norm_ac)
        rmse_err_ac = np.sqrt(rmse_err_ac/norm_ac)
        rmse_cal_ac = np.sqrt(rmse_cal_ac/norm_ac)
        
        plt.figure(figsize=(8,6))
        plt.title('Across track RMSE',fontsize=17)
        plt.plot(range(-60,60),rmse_true_ac,label='True SWOT')
        plt.plot(range(-60,60),rmse_cal_ac,label=calib+' SWOT')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('x_ac (km)',fontsize=15)
        plt.ylabel('RMSE (m)',fontsize=15)
        plt.show()

    @property
    def stats_dict(self):
        return self._d_stats
    
    
    
    def compute_psd(self, l_files, etuvar, l_files_input, lengh_scale=100, overlay=0.):
        """ compute along track psd """
            
        
        ds = xr.open_dataset(l_files)
        
        ds_input = xr.open_dataset(l_files_input)
        
        # Param of cryosat check if ok for the other sat
        delta_t = 0.9434  # s
        velocity = 6.77   # km/s
        delta_x = velocity * delta_t 
        fs_sat = 1.0 / delta_x
    
        lon_segment, lat_segment, ref_segment, true_segment, npt= compute_segment_alongtrack(ds_input.timealong, 
                                   ds_input.latalong, 
                                   ds_input.lonalong, 
                                   ds_input.refalong, 
                                   ds_input.sshalong_true, 
                                   lengh_scale,
                                   1/fs_sat,
                                   0)

        lon_segment, lat_segment, ref_segment, err_segment, npt= compute_segment_alongtrack(ds_input.timealong, 
                                       ds_input.latalong, 
                                       ds_input.lonalong, 
                                       ds_input.refalong, 
                                       ds_input.sshalong_err, 
                                       lengh_scale,
                                       1/fs_sat,
                                       0)

        lon_segment, lat_segment, ref_segment, cal_segment, npt= compute_segment_alongtrack(ds.timealong, 
                                       ds.latalong, 
                                       ds.lonalong, 
                                       ds_input.refalong, 
                                       ds[etuvar].values, 
                                       lengh_scale,
                                       1/fs_sat,
                                   0)
        
        print('Number of available segments of lengh_scale',lengh_scale,'km:',np.shape(ref_segment)[1])
        
        
        import scipy.signal

        global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                                   fs=fs_sat,
                                                                   nperseg=npt,
                                                                   scaling='density',
                                                                   noverlap=0,
                                                                   detrend=False)

        global_wavenumber, global_psd_true = scipy.signal.welch(np.asarray(true_segment).flatten(),
                                                                   fs=fs_sat,
                                                                   nperseg=npt,
                                                                   scaling='density',
                                                                   noverlap=0,
                                                                   detrend=False)

        _, global_psd_err = scipy.signal.welch(np.asarray(err_segment).flatten(),
                                                     fs=fs_sat,
                                                     nperseg=npt,
                                                     scaling='density',
                                                     noverlap=0,
                                                     detrend=False)


        _, global_psd_cal = scipy.signal.welch(np.asarray(cal_segment).flatten(),
                                             fs=fs_sat,
                                             nperseg=npt,
                                             scaling='density',
                                             noverlap=0,
                                             detrend=False)
        
        self.stats_dict['global_wavenumber'] = global_wavenumber
        self.stats_dict['global_psd_ref'] = global_psd_ref
        self.stats_dict['global_psd_true'] = global_psd_true
        self.stats_dict['global_psd_err'] = global_psd_err
        self.stats_dict['global_psd_cal'] = global_psd_cal

        global_wavenumber, global_psd_difftrue = scipy.signal.welch((np.array(ref_segment)-np.array(true_segment)).flatten(),
                                                                   fs=fs_sat,
                                                                   nperseg=npt,
                                                                   scaling='density',
                                                                   noverlap=0,
                                                                   detrend=False)
        snr_true,arr_snr_true = compute_snr1(global_psd_difftrue/global_psd_ref,global_wavenumber,0.5)

        _, global_psd_differr = scipy.signal.welch((np.array(ref_segment)-np.array(err_segment)).flatten(),
                                                     fs=fs_sat,
                                                     nperseg=npt,
                                                     scaling='density',
                                                     noverlap=0,
                                                     detrend=False)
        snr_err,arr_snr_err = compute_snr1(global_psd_differr/global_psd_ref,global_wavenumber,0.5)


        _, global_psd_diffcal = scipy.signal.welch((np.array(ref_segment)-np.array(cal_segment)).flatten(),
                                                     fs=fs_sat,
                                                     nperseg=npt,
                                                     scaling='density',
                                                     noverlap=0,
                                                     detrend=False)
        snr_cal,arr_snr_cal = compute_snr1(global_psd_diffcal/global_psd_ref,global_wavenumber,0.5)

         
        self.stats_dict['global_psd_difftrue'] = global_psd_difftrue
        self.stats_dict['global_psd_differr'] = global_psd_differr
        self.stats_dict['global_psd_diffcal'] = global_psd_diffcal
        
        self.stats_dict['snr_true'] = snr_true
        self.stats_dict['snr_err'] = snr_err
        self.stats_dict['snr_cal'] = snr_cal
        
        self.stats_dict['arr_snr_true'] = arr_snr_true
        self.stats_dict['arr_snr_err'] = arr_snr_err
        self.stats_dict['arr_snr_cal'] = arr_snr_cal
        
        

    def write_psd(self, fname, **kwargs):
        """ export des résultats vers un fichier NetCDF """
        
        
        to_write = xr.Dataset(
            data_vars=dict(
                    psd_ref=(["wavenumber"], self.stats_dict['global_psd_ref']),
                    psd_true=(["wavenumber"], self.stats_dict['global_psd_true']), 
                    psd_err=(["wavenumber"], self.stats_dict['global_psd_err']),
                    psd_cal=(["wavenumber"], self.stats_dict['global_psd_cal']), 

                    psd_difftrue=(["wavenumber"], self.stats_dict['global_psd_difftrue']),
                    psd_differr=(["wavenumber"], self.stats_dict['global_psd_differr']),
                    psd_diffcal=(["wavenumber"], self.stats_dict['global_psd_diffcal']), 

                    snr_true=( self.stats_dict['snr_true']),
                    snr_err=(self.stats_dict['snr_err']),
                    snr_cal=(self.stats_dict['snr_cal']),
                
                    arr_snr_true=(["wavenumber"], self.stats_dict['arr_snr_true']),
                    arr_snr_err=(["wavenumber"], self.stats_dict['arr_snr_err']),
                    arr_snr_cal=(["wavenumber"], self.stats_dict['arr_snr_cal']),

                ),
            coords=dict(
                wavenumber=(["wavenumber"], self.stats_dict['global_wavenumber']),
            ),
            attrs=dict(
                description=kwargs['description'] if 'description' in kwargs else 'PSD analysis',
                calib_type=kwargs['calib'] if 'calib' in kwargs else 'None',
            ),
        )
        
        to_write.to_netcdf(fname)
        
        

    def display_psd(self, fname, calib = 'calib', **kwargs):
        
        ds = xr.open_dataset(fname)

        ds = ds.assign_coords({'wavelength': 1./ds['wavenumber']})
        
        plt.figure(figsize=(7,6))
        plt.title('Power spectrum density',fontsize=17)
        plt.xlabel('Wavenumber (km$^{-1}$)',fontsize=15)
        plt.ylabel('PSD (m$^2$cy$^{-1}$km$^{-1}$)',fontsize=15)
        plt.loglog(ds.wavenumber, ds.psd_ref,'k',label='Ref',linewidth=3)
        plt.loglog(ds.wavenumber, ds.psd_true,'b--',label='True',linewidth=3)
        plt.loglog(ds.wavenumber, ds.psd_err,'r:',label='Err',linewidth=3)
        plt.loglog(ds.wavenumber, ds.psd_cal,'g',label=calib)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        
        plt.figure(figsize=(7,6))
        plt.title('Noise-to-signal ratio',fontsize=17)
        plt.xlabel('Wavenumber (km$^{-1}$)',fontsize=15)
        plt.ylabel('NSR',fontsize=15)
        plt.semilogx(ds.wavenumber, ds.psd_difftrue/ds.psd_ref,'b--',label='True') 
        plt.semilogx(1/ds.snr_true, 0.5,'bo')
        plt.semilogx(ds.wavenumber[ds.arr_snr_true], (ds.psd_difftrue/ds.psd_ref)[ds.arr_snr_true],'b',linewidth=4)

        plt.semilogx(ds.wavenumber, ds.psd_differr/ds.psd_ref,'r:',label='Err')
        plt.semilogx(ds.wavenumber[ds.arr_snr_err], (ds.psd_differr/ds.psd_ref)[ds.arr_snr_err],'r',linewidth=4)
        plt.semilogx(1/ds.snr_err, 0.5,'ro')

        plt.semilogx(ds.wavenumber, ds.psd_diffcal/ds.psd_ref,'g',label=calib) 
        plt.semilogx(ds.wavenumber[ds.arr_snr_cal], (ds.psd_diffcal/ds.psd_ref)[ds.arr_snr_cal],'g',linewidth=4)
        plt.semilogx(1/ds.snr_cal, 0.5,'go')

        plt.legend(fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot([np.min(ds.wavenumber),np.max(ds.wavenumber)],[0.5,0.5],'grey')
        plt.axis([np.min(ds.wavenumber),np.max(ds.wavenumber),0,2])
        plt.show()
        
        
        print('---------------------------------') 
        print(color.BOLD+color.RED+'Global SNR intervals'+color.END) 
        
        if np.isnan(ds.snr_true.values):
            print('SNR=0.5 SWOT true:',ds.snr_true.values,'km')
        else:
            print('SNR=0.5 SWOT true: [',ds.snr_true.values,',',1/np.min(ds.wavenumber[ds.arr_snr_true].values),'] km')

        if np.isnan(ds.snr_err.values):
            print('SNR=0.5 SWOT error:',ds.snr_err.values,'km')
        else:
            print('SNR=0.5 SWOT error: [',ds.snr_err.values,',',1/np.min(ds.wavenumber[ds.arr_snr_err].values),'] km')

        if np.isnan(ds.snr_cal.values):
            print('SNR=0.5 SWOT '+calib+':',ds.snr_cal.values,'km')
        else:
            print('SNR=0.5 SWOT '+calib+': [',ds.snr_cal.values,',',1/np.min(ds.wavenumber[ds.arr_snr_cal].values),'] km')

        
def compute_segment_alongtrack(time_alongtrack, 
                               lat_alongtrack, 
                               lon_alongtrack, 
                               ssh_alongtrack, 
                               ssh_map_interp, 
                               lenght_scale,
                               delta_x,
                               delta_t):

    segment_overlapping = 0.25
    max_delta_t_gap = 4 * np.timedelta64(1, 's')  # max delta t of 4 seconds to cut tracks

    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    #delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    if selected_track_segment.size > 0: 

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                # Near Greenwhich case
                if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                    and (lon_alongtrack[sub_segment_point] > 320.)) \
                        or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                            and (lon_alongtrack[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)


    return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 



def interp_swot2nadir(sats, dir_of_swottracks, swottracks_name='inputs', nremoval=6, ref_nadir='model'):
    
    if swottracks_name == 'inputs' :
        if np.shape(sats)[0]==1:
            readfile = os.path.isfile(dir_of_swottracks+'ref_onindepnadir.nc')
            pathfile = dir_of_swottracks+'ref_onindepnadir.nc'
        else:
            readfile = os.path.isfile(dir_of_swottracks+'ref_onnadirtracks.nc')
            pathfile = dir_of_swottracks+'ref_onnadirtracks.nc'
    
    else :
        if np.shape(sats)[0]==1:
            readfile = os.path.isfile(dir_of_swottracks+'/calib_onindepnadir.nc')
            pathfile = dir_of_swottracks+'calib_onindepnadir.nc' 
        else:
            readfile = os.path.isfile(dir_of_swottracks+'/calib_onnadirtracks.nc')
            pathfile = dir_of_swottracks+'calib_onnadirtracks.nc'
        
    
    
    if readfile: 
        
        # Reading existing files for along nadir track interpolation
        
        print('Warning: Reading from existing file:'+pathfile)
        
        ds = xr.open_dataset(pathfile)
        if swottracks_name == 'inputs' :
            return ds.lonalong, ds.latalong, ds.timealong, ds.refalong, ds.x_acalong, ds.sshalong_true, ds.sshalong_err
        else :
            return ds.lonalong, ds.latalong, ds.timealong, ds.refalong, ds.x_acalong, ds.sshalong_calib

    
    else: 
        
        # Computing along nadir track interpolation
        
        print('Computing along nadir track interpolation')
        
        import copy

        if swottracks_name == 'inputs':
            errorquad_true = []
            errorquad_err = []
        else:
            errorquad_calib = []
 

        lon = []
        lat = []
        ref_rm = []
        x_ac = []

        list_of_swottracks = sorted(os.listdir(dir_of_swottracks))
        first_iswot = None

        nstep = np.size(list_of_swottracks)
        istep = 0
        for iswot in list_of_swottracks:
            #if not iswot.startswith('dc_'):
            #    print('Warning: Some files in directory do not start by dc_*:'+iswot)
            #    continue
               
            if first_iswot is None: first_iswot=iswot
                
            ds_swot = xr.open_mfdataset(dir_of_swottracks+iswot)
            #print(iswot)
            progress_bar(istep,nstep-1)
            for isat in sats: 
                ds_sat = xr.open_dataset(isat)

                timedelay = np.timedelta64(12,'h') 
                mintime,maxtime = np.array(np.mean(ds_swot.time), dtype='datetime64[h]')-timedelay,np.array(np.mean(ds_swot.time), dtype='datetime64[h]')+timedelay

                ds_sat_cut = ds_sat.sel({'time':slice(mintime,maxtime)})

                if ds_sat_cut.time.size != 0:  

                    if first_iswot == iswot and sats[0]==isat:
                        time = np.array(ds_sat_cut.time)
                    else:
                        time = np.hstack((time,np.array(ds_sat_cut.time)))

                    lon = np.hstack((lon,np.array(ds_sat_cut.lon)))
                    lat = np.hstack((lat,np.array(ds_sat_cut.lat)))
                    
                    swath1 = np.arange(10,60,1.45)
                    swath2 = - swath1[::-1]
                    swot_swaths = np.hstack((swath1,[np.nan],swath2))
                    x_ac_swot = ds_swot.ssh_true/ds_swot.ssh_true
                    for ial in range(np.shape(x_ac_swot)[0]):
                        x_ac_swot[ial,:] = swot_swaths
                         

                    #x_ac_out = interpolate.griddata((np.array(ds_swot.longitude).ravel(),np.array(ds_swot.latitude).ravel()),
                    #           np.array(ds_swot.ssh_true).ravel(),
                    #           (ds_sat_cut.lon,ds_sat_cut.lat))

                    x_ac = np.hstack((x_ac,ds_sat_cut.lon)) 

                    if swottracks_name == 'inputs':
                        var_out = interpolate.griddata((np.array(ds_swot.longitude).ravel(),np.array(ds_swot.latitude).ravel()),
                                   np.array(ds_swot.ssh_true).ravel(),
                                   (ds_sat_cut.lon,ds_sat_cut.lat)) 
                        errorquad_true = np.hstack((errorquad_true,var_out)) 


                        var_out = interpolate.griddata((np.array(ds_swot.longitude).ravel(),np.array(ds_swot.latitude).ravel()),
                                   np.array(ds_swot.ssh_err).ravel(),
                                   (ds_sat_cut.lon,ds_sat_cut.lat))

                        errorquad_err = np.hstack((errorquad_err,var_out))

                        eq_true_rm = copy.deepcopy(errorquad_true)
                        eq_err_rm = copy.deepcopy(errorquad_err) 


                    else:  
                        var_out = interpolate.griddata((np.array(ds_swot.longitude).ravel(),np.array(ds_swot.latitude).ravel()),
                                   np.array(ds_swot[swottracks_name]).ravel(),
                                   (ds_sat_cut.lon,ds_sat_cut.lat))  
                        errorquad_calib = np.hstack((errorquad_calib,var_out)) 

                        eq_calib_rm = copy.deepcopy(errorquad_calib)

                    if ref_nadir == 'model':
                        ref_rm = np.hstack((ref_rm,np.array(ds_sat_cut.sla_unfiltered*var_out/var_out)))
                    if ref_nadir == 'obs':
                        ref_rm = np.hstack((ref_rm,np.array(ds_sat_cut.ssh_obs*var_out/var_out)))

            istep += 1

        if nremoval !=0:
            if swottracks_name == 'inputs':
                for i in range(np.shape(x_ac)[0]): 
                    if x_ac[i]<-60+nremoval: 
                        eq_true_rm[i] = np.nan 
                        eq_err_rm[i] = np.nan  
                    if x_ac[i]>60-nremoval: 
                        eq_true_rm[i] = np.nan  
                        eq_err_rm[i] = np.nan 
                    if x_ac[i]>-10-nremoval and x_ac[i]<10+nremoval: 
                        eq_true_rm[i] = np.nan  
                        eq_err_rm[i] = np.nan 
            else:
                for i in range(np.shape(x_ac)[0]): 
                    if x_ac[i]<-60+nremoval: 
                        eq_calib_rm[i] = np.nan  
                    if x_ac[i]>60-nremoval: 
                        eq_calib_rm[i] = np.nan   
                    if x_ac[i]>-10-nremoval and x_ac[i]<10+nremoval: 
                        eq_calib_rm[i] = np.nan  


        if swottracks_name == 'inputs':

            ds = xr.Dataset(
                 data_vars=dict(
                     lonalong=(["time"], lon),
                     latalong=(["time"], lat), 
                     timealong=(["time"], time),
                     refalong=(["time"], ref_rm),
                     x_acalong=(["time"], x_ac),
                     sshalong_true=(["time"], eq_true_rm),
                     sshalong_err=(["time"], eq_err_rm)
                 ),
                 coords=dict(
                     time=(["time"], np.arange(np.size(lon))) 
                 ),
                 attrs=dict(description="Data challenge: https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS. \nCreated with interp_swot2nadir() in eval_on_nadirtrack.py."),
             )        

            if np.shape(sats)[0]==1:
                ds.to_netcdf(dir_of_swottracks+'/ref_onindepnadir.nc')
            else:
                ds.to_netcdf(dir_of_swottracks+'/ref_onnadirtracks.nc')

            return lon, lat, time, ref_rm, x_ac, eq_true_rm, eq_err_rm
        else: 

            ds = xr.Dataset(
                 data_vars=dict(
                     lonalong=(["time"], lon),
                     latalong=(["time"], lat), 
                     timealong=(["time"], time),
                     refalong=(["time"], ref_rm),
                     x_acalong=(["time"], x_ac),
                     sshalong_calib=(["time"], eq_calib_rm) 
                 ),
                 coords=dict(
                     time=(["time"], np.arange(np.size(lon))) 
                 ),
                 attrs=dict(description="Data challenge: https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS. \nCreated with interp_swot2nadir() in eval_on_nadirtrack.py."),
             )        

            if np.shape(sats)[0]==1:
                ds.to_netcdf(dir_of_swottracks+'/calib_onindepnadir.nc')
            else:
                ds.to_netcdf(dir_of_swottracks+'/calib_onnadirtracks.nc')

            return lon, lat, time, ref_rm, x_ac, eq_calib_rm
    
    
    
    
    
def compute_snr1(array, wavenumber, threshold=0.5):
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

        if len(list_of_res) > 0:
            resolution_scale = np.nanmin(np.asarray(list_of_res))
        else: 
            resolution_scale = np.nanmin(1./wavenumber[wavenumber!=0])

    else:  
        if np.all( array - threshold>0 )>0:
            resolution_scale = np.nan
        else : 
            resolution_scale = np.nanmin(1./wavenumber[wavenumber!=0])

    array_resol_scales = array - threshold<0
    #print(list_of_res) 

    return resolution_scale, array_resol_scales



def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)
    

        
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
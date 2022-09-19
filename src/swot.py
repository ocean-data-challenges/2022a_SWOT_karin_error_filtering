import xarray as xr
import hvplot.xarray
import numpy as np
import sys
import matplotlib.pylab as plt
sys.path.append('..')
from src.filters_bidim import median_filter, lanczos_filter, loess_filter, gaussian_filter, boxcar_filter, lee_filter


class SwotTrack(object):

    def __init__(self, fname=None, dset=None):
        """ constructeur """
        if fname is not None:
            self._fname = fname
            self._dset = xr.open_dataset(self.filename)
        elif dset is not None:
            self._fname = None
            self._dset = dset
        else:
            raise Exception('either fname or dset should be provided')

        self._nadir_mask = None
        
        
    
    def compute_geos_current(self, invar, outvar):
        
        def coriolis_parameter(lat):  
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
        
        ds = self._dset
        
        dx = 2000 # m
        dy = 2000 # m
        gravity = 9.81
        f_coriolis = coriolis_parameter(ds.latitude.values)
        ref_gx, ref_gy = gravity/f_coriolis*np.gradient(ds[invar], dx, edge_order=2)
        geos_current = np.sqrt(ref_gx**2 + ref_gy**2)
        
        self.fc = f_coriolis
        
        self.__enrich_dataset(outvar, geos_current)
        self.__enrich_dataset(outvar + '_x', ref_gx)
        self.__enrich_dataset(outvar + '_y', ref_gy)
        
        
    def compute_relative_vorticity(self, invar_x, invar_y, outvar):
        
        ds = self._dset
        
        dx = 2000 # m
        dy = 2000 # m
        
        du_dx, du_dy = np.gradient(ds[invar_x], dx, edge_order=2)
        dv_dx, dv_dy = np.gradient(ds[invar_y], dx, edge_order=2)
        
        ksi = (dv_dx - du_dy)/self.fc
        
        self.__enrich_dataset(outvar, ksi)
        
        
    def display_demo_target(self):
        
        ds = self._dset
        ds = ds.isel(num_lines=slice(2400, 3000), drop=True)
        ds['num_lines'] = 2*(ds['num_lines']-ds['num_lines'][0])
        ds['num_pixels'] = 2*ds['num_pixels']
        
        msk = ds.simulated_noise_ssh_karin/ds.simulated_noise_ssh_karin
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.subplot(311)
        (ds.simulated_true_ssh_karin*msk).T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('TARGET: SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.subplot(312)
        (ds.simulated_true_geos_current*msk).T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('TARGET: Geos. current from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.subplot(313)
        vmin = np.nanpercentile(ds['simulated_true_ksi'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ksi'], 95)
        vdata = np.maximum(np.abs(vmin), np.abs(vmax))
        (ds.simulated_true_ksi*msk).T.plot(vmin=-vdata, vmax=vdata, cmap='BrBG', cbar_kwargs={'label': '[s$^{-1}$]'})
        plt.title('TARGET: Relative vorticity from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        
        plt.show()
        

    def display_demo(self, var_name='karin',msk=None, vmin=None, vmax=None):
        
        ds = self._dset
        ds = ds.isel(num_lines=slice(2400, 3000), drop=True)
        ds['num_lines'] = 2*(ds['num_lines']-ds['num_lines'][0])
        ds['num_pixels'] = 2*ds['num_pixels']
        
        if msk is None:
            msk = ds['ssh_'+var_name]/ds['ssh_'+var_name]
        if vmin is None:
            vmin = np.nanpercentile(ds['ssh_'+var_name], 5)
        if vmax is None:
            vmax = np.nanpercentile(ds['ssh_'+var_name], 95)
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.subplot(311)
        (ds['ssh_'+var_name]*msk).T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('SSH '+var_name, fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        #plt.xlim(2400, 3000)
        plt.subplot(312)
        (ds['geos_current_'+var_name]*msk).T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from SSH '+var_name, fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(313)
        vmin_ksi = np.nanpercentile(ds['ksi_'+var_name]*msk, 5)
        vmax_ksi = np.nanpercentile(ds['ksi_'+var_name]*msk, 95)
        vdata = np.maximum(np.abs(vmin_ksi), np.abs(vmax_ksi))
        (ds['ksi_'+var_name]*msk).T.plot(vmin=-vdata, vmax=vdata, cmap='BrBG', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Relative vorticity from Geos. current '+var_name, fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        #plt.xlim(2400, 3000)
        
        plt.show()
        
        return msk,vmin, vmax
        
        
    
    def display_demo_input(self):
        
        ds = self._dset
        ds = ds.isel(num_lines=slice(2400, 3000), drop=True)
        ds['num_lines'] = 2*(ds['num_lines']-ds['num_lines'][0])
        ds['num_pixels'] = 2*ds['num_pixels']
        
        msk = ds.simulated_noise_ssh_karin/ds.simulated_noise_ssh_karin
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        plt.subplot(311)
        ds.simulated_noise_ssh_karin.T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('INPUT: SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')

        plt.subplot(312)
        ds.simulated_noisy_geos_current.T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.subplot(313)
        vmin = np.nanpercentile(ds['simulated_true_ksi'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ksi'], 95)
        vdata = np.maximum(np.abs(vmin), np.abs(vmax))
        (ds.simulated_noisy_ksi*msk).T.plot(vmin=-vdata, vmax=vdata, cmap='BrBG', cbar_kwargs={'label': '[s$^{-1}$]'})
        plt.title('TARGET: Relative vorticity from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')

        
        plt.show()
        
        
    def display_result_quickstart(self):
        
        ds = self._dset
        ds = ds.isel(num_lines=slice(2400, 3000), drop=True)
        ds['num_lines'] = 2*(ds['num_lines']-ds['num_lines'][0])
        ds['num_pixels'] = 2*ds['num_pixels']
        
        msk = ds.simulated_noise_ssh_karin/ds.simulated_noise_ssh_karin
        vmin = np.nanpercentile(ds['simulated_true_ssh_karin'], 5)
        vmax = np.nanpercentile(ds['simulated_true_ssh_karin'], 95)
        
        plt.figure(figsize=(22, 12))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.4)
        
        plt.subplot(421)
        (ds.simulated_true_ssh_karin*msk).T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('TARGET: SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(422)
        (ds.simulated_true_geos_current*msk).T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('TARGET: Geos. current from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(423)
        ds.simulated_noise_ssh_karin.T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('INPUT: SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(424)
        ds.simulated_noisy_geos_current.T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(425)
        ds.ssh_karin_filt.T.plot(vmin=vmin, vmax=vmax, cmap='Spectral_r', cbar_kwargs={'label': '[m]'})
        plt.title('RESULT: Filtered SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        plt.subplot(426)
        ds.geos_current_ssh_karin_filt.T.plot(vmin=0, vmax=0.5, cmap='Blues_r', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from Filtered SSH true + KaRin noise', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        data = ds.ssh_karin_filt.T - (ds.simulated_true_ssh_karin*msk)
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 95)
        maxval = np.maximum(vmin, vmax)
        plt.subplot(427)
        data.plot(vmin=-maxval, vmax=maxval, cmap='coolwarm', cbar_kwargs={'label': '[m]'})
        plt.title('Filtered SSH true + KaRin noise - SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        data = (ds.geos_current_ssh_karin_filt.T - (ds.simulated_true_geos_current*msk).T)
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 95)
        maxval = np.maximum(vmin, vmax)
        plt.subplot(428)
        data.plot(vmin=-maxval, vmax=maxval, cmap='coolwarm', cbar_kwargs={'label': '[m.s$^{-1}$]'})
        plt.title('Geos. current from Filtered SSH true + KaRin noise - Geos. current from SSH true', fontweight='bold')
        plt.xlabel('[km]')
        plt.ylabel('[km]')
        
        plt.show()
        
    
        
        
    def plot_track(self,filtered_var_name,swottrack_input):
        
        ds = self._dset
        ds0 = swottrack_input._dset 
        
        vmin = np.nanpercentile(ds0['ssh_karin'], 5)
        vmax = np.nanpercentile(ds0['ssh_karin'], 95)
         
        
        fig = plt.figure(figsize=(18,12))
        ax1 = fig.add_subplot(2,3,1)        
        plt.scatter(ds0.longitude,ds0.latitude,c=ds0['ssh_true'].values, vmin= vmin, vmax= vmax, cmap='Spectral_r')
        plt.colorbar()
        ax1.title.set_text('True ssh')
                   
        ax2 = fig.add_subplot(2,3,2)        
        plt.scatter(ds0.longitude,ds0.latitude,c=ds0['ssh_karin'].values, vmin= vmin, vmax= vmax, cmap='Spectral_r')
        plt.colorbar()
        ax2.title.set_text('Noisy ssh karin')
                   
        ax3 = fig.add_subplot(2,3,3)        
        plt.scatter(ds.longitude,ds.latitude,c=ds[filtered_var_name].values, vmin= vmin, vmax= vmax, cmap='Spectral_r')
        plt.colorbar()
        ax3.title.set_text('Filtered ssh karin')
         
    
        delta0 = ds0['ssh_karin'] - ds0['ssh_true']
        delta = ds[filtered_var_name] - ds0['ssh_true']
        
        vmin_delta = np.nanpercentile(delta.values, 5)
        vmax_delta = np.nanpercentile(delta.values, 95)
                   
        ax5 = fig.add_subplot(2,3,5)        
        plt.scatter(ds0.longitude,ds0.latitude,c=delta0, vmin= vmin_delta, vmax= vmax_delta, cmap='bwr')
        plt.colorbar()
        ax5.title.set_text('Karin noise')
                   
        ax6 = fig.add_subplot(2,3,6)        
        plt.scatter(ds.longitude,ds.latitude,c=delta, vmin= vmin_delta, vmax= vmax_delta, cmap='bwr')
        plt.colorbar()
        ax6.title.set_text('Filtered ssh karin - True ssh karin')
                   
                   
        plt.show()
        
        
    def display_track(self):
        
        ds = self._dset
        
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

        
    def apply_your_own_filter(self,thefilter,invar,outvar,**kwargs):
        """ apply median filter, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = thefilter(ssha, **kwargs)
        self.__enrich_dataset(outvar, ssha_f)

        
    def apply_median_filter(self, invar, size, outvar):
        """ apply median filter, enrich dataset inplace """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = median_filter(ssha, size=size)
        self.__enrich_dataset(outvar, ssha_f)

    def apply_gaussian_filter(self, invar, sigma, outvar):
        """ apply gaussian filter """
        self.__check_var_exist(invar)
        if outvar in self.dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = gaussian_filter(ssha, sigma)
        self.__enrich_dataset(outvar, ssha_f)

    def apply_boxcar_filter(self, invar, size, outvar):
        """ apply boxcar filter """
        self.__check_var_exist(invar)
        if outvar in self.dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = boxcar_filter(ssha, size)
        self.__enrich_dataset(outvar, ssha_f)

    def apply_lanczos_filter(self, invar, lx, outvar, width_factor=3):
        """ apply a lanczos filter """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = lanczos_filter(ssha, lx, width_factor=width_factor)
        self.__enrich_dataset(outvar, ssha_f)

    def apply_loess_filter(self, invar, outvar, deg=1, l=1, kernel='gaussian'):
        """ apply a loess filter """
        self.__check_var_exist(invar)
        if outvar in self._dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        ssha_f = loess_filter(ssha, deg, l, kernel)
        self.__enrich_dataset(outvar, ssha_f)
        
    def apply_lee_filter(self, invar, lx, outvar):
        ''' apply filter from lee et al., 1980 '''
        self.__check_var_exist(invar)
        if outvar in self.dset.data_vars:
            self._dset = self._dset.drop(outvar)
        ssha = self.dset[invar].values
        sshaf = lee_filter(ssha, lx)
        self.__enrich_dataset(outvar, sshaf)
        
        
    def to_netcdf(self, l_variables, fname):
        """ write to netcdf file """
        if l_variables == 'all':
            l_variables = list(self.dset.data_vars)
        if 'time' not in l_variables:
            l_variables.append('time')

        to_write = self.dset[l_variables]
        to_write.to_netcdf(fname)

    def __check_var_exist(self, varname):
        """ checks for the availability of a variable in the dataset """
        if varname not in self._dset.data_vars:
            raise Exception('variable %s is not defined' % varname)

    def add_swath_variable(self, varname, array, replace=False):
        """ add a 2d variable to the dataset """
        
        if varname in self._dset.data_vars:
            if replace:
                self._dset = self._dset.drop(varname)
            else:
                raise Exception('variable %s already exists' %varname)
        self.__enrich_dataset(varname, array)

    def __enrich_dataset(self, varname: str,  array) -> None:
        """ add a new variable to the dataset """
        self._dset = self._dset.assign(dict(temp=(('num_lines', 'num_pixels'), array)))
        self._dset = self._dset.rename_vars({'temp': varname})

    def fill_nadir_gap(self, invar):
        """ fill nadir gap """
        self.__check_var_exist(invar)
        ssha = self.dset[invar].values
        self._nadir_mask = np.isnan(ssha)

        at = pyinterp.Axis(self.x_at)
        ac = pyinterp.Axis(self.x_ac)
        grid = pyinterp.grid.Grid2D(at,ac,ssha)
        has_converged, filled = gauss_seidel(grid)

        if has_converged:
            self.dset[invar][:] = filled
        else:
            raise Exception('nadir gap filling failed')

    def empty_nadir_gap(self, invar):
        """ empty nadir gap by applying the mask back on """
        self.__check_var_exist(invar)
        if self.nadir_mask is not None:
            self.dset[invar].values[self.nadir_mask] = np.nan
            
            
    
    @property
    def nadir_mask(self):
        """ return nadir mask """
        return self._nadir_mask

    @property
    def x_at(self):
        return np.arange(self.dset.dims['num_lines'])

    @property
    def x_ac(self):
        return np.arange(self.dset.dims['num_pixels'])

    @property
    def longitude(self):
        return self.dset.longitude.values

    @property
    def latitude(self):
        return self.dset.latitude.values

    @property
    def cycle(self):
        return self.dset.attrs['cycle_number']

    @property
    def track(self):
        return self.dset.attrs['pass_number']

    @property
    def filename(self):
        return self._fname

    @property
    def nx(self):
        return self._dset.dims['num_lines']

    @property
    def ny(self):
        return self._dset.dims['num_pixels']

    @property
    def dset(self):
        return self._dset

    @property
    def minlon(self):
        return np.min(self._dset.longitude.values)

    @property
    def maxlon(self):
        return np.max(self._dset.longitude.values)

    @property
    def minlat(self):
        return np.min(self._dset.latitude.values)

    @property
    def maxlat(self):
        return np.max(self._dset.latitude.values)





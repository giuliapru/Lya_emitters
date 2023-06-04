import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import scipy.constants as const


#####################
#FUNCTIONS#
#####################

def Length_box(lim_inf, lim_sup):
    """ Length of the squared box [cMpc] """
    return np.abs(lim_sup-lim_inf)

def total_volume(lim_inf, lim_sup):
    """ Volume of the squared box [cMpc^3] """
    return np.abs((Length_box(lim_inf, lim_sup))**3)

def corr_func(Nrand, bins, Pos, lim_inf, lim_sup, verb):
    """    
    Parameters:
    -----------------
    Nrand   : int
        The number of random galaxies of which it computes the density profile
    bins    : numpy array (1D)
        The bins in cMpc of projected distance
    Pos     : numpy array (3D)
        The first two entries must be the projected positions of all the galaxies to consider
    lim_inf : float 
        The lower limit of the box we want to consider (useful if we are testing this function)
    lim_sup : float 
        The higher limit of the box we want to consider
    verb    : bool
        If True it prints some indications on the processes (useful to check the time it takes for each part)
        
    Returns:
    -----------------
    bins_cen : numpy array
        The center of the bins in cMpc
    xi       : numpy array
        The values of the correlation function
    err_xi   : numpy array
        Poisson error on xi, not really useful if you want to extimate uncertainties via bootstrap
    """
    
    #selecting only galaxies in the box delimited by lim_inf and lim_sup
    if lim_inf != 0 and lim_sup != 100:
        ind = []
        for i in range(len(Pos)):
            if lim_inf<Pos[i,0]<lim_sup and lim_inf<Pos[i,1]<lim_sup and lim_inf<Pos[i,2]<lim_sup:
                ind.append('True')
            else:
                ind.append('False')
        ind = np.array(ind)
        Pos = Pos[ind=='True'][:]
    else:
        Pos = Pos
    
    random_gal = np.random.randint(0, len(Pos), Nrand) #random galaxies indices
    L_box = Length_box(lim_inf, lim_sup)
    tot_vol = total_volume(lim_inf, lim_sup)
    
    Ngal = len(Pos)
    if verb==True:
        print('The number of galaxies in side the considered volume is: ', Ngal)
    
    Nbins = len(bins)-1                                        
    bins_cen = (bins[1:]+bins[:-1])/2 
    dvol = 4/3*np.pi*(bins[1:]**3 - bins[:-1]**3)
    
    if verb==True:
        print('Evaluating distances...')
        
    #distances between the selected random galaxies and all the others
    dd = np.zeros((Nrand,len(Pos)))
    for i,j in enumerate(random_gal):
        delta_c = np.abs(Pos[j] - Pos)
        delta_c = np.where(delta_c>L_box/2, np.abs(delta_c -L_box), delta_c) #boundary conditions
        dd[i] = np.sqrt(np.sum(delta_c**2, axis=1))
    
    if verb ==True:
        print('Counting elements inside the bins...')
        
    #number of galaxies in each bin
    dens_prof = np.zeros((Nrand, Nbins))
    for i in range(Nrand):
        dens_prof[i], _=np.histogram(dd[i], bins=bins)
    
    #normalizing by the volume
    dens_prof = dens_prof/dvol
    
    #average density profile
    av_dens_prof = np.mean(dens_prof, axis=0)
    err_dens_prof = 1/Nrand*np.sqrt(np.sum((dens_prof/dvol), axis=0)) #poisson error
    
    #getting xi as the mean density for each bin over the total average density - 1
    xi = av_dens_prof/(Ngal/tot_vol) -1
    err_xi = err_dens_prof/(Ngal/tot_vol)
    
    return bins_cen, xi, err_xi


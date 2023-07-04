import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import scipy.constants as const

def Lya_lumfunction(L_lya, Nbins, len_box):
    
    minimo = np.min(L_lya[L_lya!=0]) #minimum Lya_lum 
    massimo = np.max(L_lya) #maximum Lya_lum
    
    Ngal_tot = len(L_lya)
    Ngal_lya = len(L_lya[L_lya!=0])
    tot_vol = len_box**3
    
    lyabins = np.logspace(np.log10(minimo), np.log10(massimo), Nbins)   
    lyabins_average_log = np.array((np.log10(lyabins[1:]) - np.log10(lyabins[:-1]))/2 + np.log10(lyabins[:-1]))
    lyabins_average = 10**lyabins_average_log
    
    counts, _ = np.histogram(L_lya[L_lya!=0], bins=lyabins)
    
    counts_norm = counts/(tot_vol*(np.log10(lyabins[1:])-np.log10(lyabins[:-1])))
    err_norm = np.sqrt(counts)/(tot_vol*(np.log10(lyabins[1:])-np.log10(lyabins[:-1])))
    #check
    #if np.sum(counts)==Ngal_lya:
    #    print('all galaxies in!')
    
    return lyabins_average, counts_norm, err_norm



def lum_func_mag(mag, Nbins, len_box):
    
    minimo = np.min(mag) #minimum magnitude in the sample
    massimo = np.max(mag) #maximum magnitude in the sample
    
    Ngal_tot = len(mag)
    tot_vol = len_box**3
    
    bins = np.linspace(minimo, massimo, Nbins)   
    bins_average = np.array((bins[1:] - bins[:-1])/2 + bins[:-1])
    
    counts, _ = np.histogram(mag, bins=bins)
    
    counts_norm = counts/(tot_vol*(bins[1:]-bins[:-1]))
    err_norm = np.sqrt(counts)/(tot_vol*(bins[1:]-bins[:-1]))
    
    return bins_average, counts_norm, err_norm
    
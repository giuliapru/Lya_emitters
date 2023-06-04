import numpy as np
import astropy.stats
from scipy import stats
import matplotlib as mp
import astropy.units as u
import matplotlib.cm as cm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
import matplotlib.ticker as ticker
from astropy.modeling import fitting, models
from astropy.cosmology import Planck15 as cosmo
from astropy.modeling.models import custom_model

def vsep(zcent, z1):	 	# Proper LOS distance [pMpc]
	_c      = const.c/1e3	# km/s
	delta_z = (z1-zcent)/(1+zcent)
	delta_v = delta_z * _c
	return np.abs(delta_v)/cosmo.H(np.mean(np.array([z1,zcent]))).value

def dang(ra,dec,z,ra_cent,dec_cent,z_cent):	# Angular Projected Distance to Proper [pMpc]
	delta_dec = dec-dec_cent
	delta_ra  = (ra-ra_cent)*np.cos(np.radians(dec_cent))
	kpcarc    = cosmo.kpc_proper_per_arcmin((z+z_cent)*.5).value*60.
	ang_dist  = np.sqrt(delta_dec**2+delta_ra**2) * kpcarc / 1000
	return ang_dist	

# Below the function I used to fit the projected correlation function
# You may need to adapt the parameters depending on your science
@custom_model
def wpfunc8(RR, R0=4):
	gamma = 1.8
	dv    = 500
	zcen  = np.median(zLAE)
	zup   = ((1+zcen)*dv/2.99792e5)+zcen
	z0    = (cosmo.comoving_distance(zup)-cosmo.comoving_distance(zcen)).value*cosmo.h
	hyp   = hyp2f1(0.5,gamma/2.,1.5,-z0**2/RR**2)
	return (R0/RR)**gamma * hyp

@custom_model
def wpfunc5(RR,R0=4):
	gamma = 1.5
	dv    = 500
	zcen  = np.median(zLAE)
	zup   = ((1+zcen)*dv/2.99792e5)+zcen
	z0    = (cosmo.comoving_distance(zup)-cosmo.comoving_distance(zcen)).value*cosmo.h
	hyp   = hyp2f1(0.5,gamma/2.,1.5,-z0**2/RR**2)
	return (R0/RR)**gamma * hyp

# Global Definitions
_c = const.c/1e3	# light speed in km/s

# Matplotlib Setup
mp.rc('text', usetex=True)
mp.rcParams['axes.labelsize']  = 15
mp.rcParams['axes.titlesize']  = 15
mp.rcParams['xtick.labelsize'] = 13
mp.rcParams['ytick.labelsize'] = 13

###########################
# Read Galaxies Catalogue
###########################
# Load here your catalogue and the variables you need
# e.g. sky coordinates (RA,DEC) in decimal degrees
# You will need to change these lines depending on the keyword in your catalogue
LAE     = fits.open('XXX')[1].data
ra      = LAE['RA_deg']
dec     = LAE['DEC_deg']
zLAE    = LAE['redshift']

###################################
# Define Distances
###################################
# Define the binning in LOS direction
vmax   = 3000
Nvbin  = 10	
# Convert LOS separation from km/s to pMpc
vmin   = 50/cosmo.H(np.mean(zLAE)).value
vlim   = vmax/cosmo.H(np.mean(zLAE)).value
# Define log-spaced intervals of distance
dvbin  = np.logspace(np.log10(vmin),np.log10(vlim),Nvbin+1)
# Add a larger bin at the beginning of the array
# where you do not expect to find many sources
dvbin  = np.insert(dvbin,0,5/cosmo.H(np.mean(zLAE)).value)
dvbin_cent = (dvbin[1:]+dvbin[:-1])/2.	# pMpc
xerr   = (dvbin[1:]-dvbin[:-1])
# Upper Integration Limit coinverted from a LOS velocity in km/s into [pMpc]
vuplim = 500/cosmo.H(np.mean(zLAE)).value	
Nvbin += 1		# Final number of bin

# Define the binning in transverse direction
dbin  = np.array([2/60*cosmo.kpc_comoving_per_arcmin(3.).value*cosmo.h/1000,0.15,0.25,0.35,0.45,0.60])
dbin_cent = (dbin[1:]+dbin[:-1])/2		# [Mpc/h]
Ndbin = len(dbin)-1	# Final number of bin

#################
# DATA Galaxies
#################
# Data-Data Pairs
skip = []
# We will save, for wach pair, the LOS separation (centered on the ii-th LAE), transverse separation, mean redshift of the pair
dv_DD, d_DD, z_DD = ([] for k in range(3))
print('Generating DD Pairs ...')

for ii in range(len(zLAE)):
	# Keep track of the LAEs already used
	# to avoid a data-data pair to be built with the same galaxy taken twice	
	skip.append(ii)		
	for jj in range(len(zLAE)):			
		if jj not in skip:
			dv_DD.append(vsep(zLAE[ii],zLAE[jj]))
			d_DD.append(dang(ra[jj],dec[jj],zLAE[jj],ra[ii],dec[ii],zLAE[ii]))
			z_DD.append(np.nanmean(np.array([zLAE[jj],zLAE[ii]])))

z_DD  = np.array(z_DD)
d_DD  = np.asarray(d_DD)
dv_DD = np.asarray(dv_DD)
# Total number of Pairs
nDD   = len(zLAE)*(len(zLAE)-1)/2

'''
A quick comment: 
In my paper I have LAEs from 28 different field (don't know if it's your case as well, but may be useful). When I do the pairs, it's not required that the two LAEs are in the same field. It's not necessary because the field are very far once from another: the LOS distance maybe small if they have similar redshift, but the transverse distance is large and they will be authomatically excluded from the significant interval I'll consider to compute the projected correlation function.
'''

#####################
# RANDOM Catalogue
#####################
# Bootstrap + Projecting the 2D function
Nboot    = 100	
R0boot8  = np.zeros(Nboot)
R0boot5  = np.zeros(Nboot)
fGG_boot = np.zeros((Nboot,Ndbin,Nvbin))
wGG_boot = np.zeros((Nboot,Ndbin))

'''
You may need to edit the following lines. 
I had a catalogue of random sources generated with a spefic method. Since it's big, it takes ages to do the random-random pairs. I've run the code on a cluster and save the random-random LOS and trasverse separation in separated files and did the experiment many times (~100) since it's random and the result, in principle, may change every time. In this case I only load the pre-compiled catalogue for the random-random pair and simply compute the 2D correlation function.
'''

print('Starting Bootstrap ...')
for kk in range(Nboot):	

	# Load pre-compiled RR pairs results
	data  = ascii.read(basefold+'fGG_dist/boot_{}.csv'.format(kk+1))
	d_RR  = data['d_RR']
	dv_RR = data['dv_RR']
	z_RR  = data['zemi_RR'] 
	nRR   = len(d_RR)

	# Moving to Co-Moving distance in [Mpc/h]
	if kk == 0: # First loop only!
		d_DD = d_DD*(1+z_DD)*cosmo.h
	d_RR = d_RR*(1+z_RR)*cosmo.h

	# Counting how many pairs in each 2D bin of distances
	DD = np.zeros((Ndbin,Nvbin))
	RR = np.zeros((Ndbin,Nvbin))

	for ii in range(Ndbin):		# Transverse Distance
		dtrans      = np.where( (d_DD>=dbin[ii])&(d_DD<dbin[ii+1]) )[0]
		rand_dtrans = np.where( (d_RR>=dbin[ii])&(d_RR<dbin[ii+1]) )[0]

		for jj in range(Nvbin):	# Line-of-sight
			dlos      = np.where( (dv_DD[dtrans]>=dvbin[jj])&(dv_DD[dtrans]<dvbin[jj+1]) )[0]
			rand_dlos = np.where( (dv_RR[rand_dtrans]>=dvbin[jj])&(dv_RR[rand_dtrans]<dvbin[jj+1]) )[0]
			DD[ii,jj] = len(dlos)
			RR[ii,jj] = len(rand_dlos)

	# Compute the 2D correlation function for this random realization
	fGG_boot[kk] = ((DD/nDD)/(RR/nRR))-1

	###################################
	# PROJECTED Cross-Correlation 
	# -> Integrate over LOS Separation
	###################################
	for ii in range(Ndbin):
		for jj in range(Nvbin):
			if dvbin_cent[jj]<=vuplim:
				wGG_boot[kk,ii] += fGG_boot[kk,ii,jj]*xerr[jj]	# xerr=dz, size of the current LOS bin
	print('{}/{}'.format(kk+1,Nboot), wGG_boot[kk])				# It cahnges in every bin as they're log-spaced and not equally-spaced!

	###########################
	# Fit with Power-Law Model 
	###########################
	# Gamma=1.8
	wpmodel     = wpfunc8(R0=4.)
	solver      = fitting.LevMarLSQFitter()
	results     = solver(wpmodel, dbin_cent, wGG_boot[kk,:])
	R0boot8[kk] = results.R0.value
	# Gamma=1.5
	wpmodel     = wpfunc5(R0=4.)
	solver      = fitting.LevMarLSQFitter()
	results     = solver(wpmodel, dbin_cent, wGG_boot[kk,:])
	R0boot5[kk] = results.R0.value
		
###################################
# Finalizing the Bootstrap Procedure
###################################
# We have done the experiment above many times, for many different random saples
# Pur final result is not a single number, but a distribution
# Let's take our fiducal estimate from the median and the percentiles (to get the uncertainty) of the distribution
wGG       = np.zeros(Ndbin)
wGG_sigma = np.zeros(Ndbin)
fGG       = np.zeros((Ndbin,Nvbin))
fGG_sigma = np.zeros((Ndbin,Nvbin))
for ii in range(Ndbin):
	wGG[ii]       = np.median(wGG_boot[:,ii])
	wGG_sigma[ii] = np.abs((np.percentile(wGG_boot[:,ii],84)-np.percentile(wGG_boot[:,ii],16))/2)
	for jj in range(Nvbin):
		fGG[ii,jj]       = np.median(fGG_boot[:,ii,jj])
		fGG_sigma[ii,jj] = np.abs((np.percentile(fGG_boot[:,ii,jj],84)-np.percentile(fGG_boot[:,ii,jj],16))/2)

###################################
# Final Fit with Power-Law Model 
###################################
# Gamma=1.8
wpmodel   = wpfunc8(R0=4.)
solver    = fitting.LevMarLSQFitter()
results   = solver(wpmodel, dbin_cent, wGG, weights=1./wGG_sigma**2)
R0sigma8  = np.array([np.percentile(R0boot8,84)-np.percentile(R0boot8,50),np.percentile(R0boot8,50)-np.percentile(R0boot8,16)])
R0best8   = results.R0.value
print('R0 best: {} + {} - {}'.format(R0best8,R0sigma8[0],R0sigma8[1]))

# Gamma=1.5
wpmodel   = wpfunc5(R0=4.)
solver    = fitting.LevMarLSQFitter()
results   = solver(wpmodel, dbin_cent, wGG, weights=1./wGG_sigma**2)
R0sigma5  = np.array([np.percentile(R0boot5,84)-np.percentile(R0boot5,50),np.percentile(R0boot5,50)-np.percentile(R0boot5,16)])
R0best5   = results.R0.value
print('R0 best: {} + {} - {}'.format(R0best5,R0sigma5[0],R0sigma5[1]))

# At this point, if this code takes too long to run, I'd save the results in a file
# so that it's easier to access the output and do figures without waiting a few hours every time just to change a color!

###################################
# Diagnostic Figure
###################################
fitline  = np.linspace(np.min(dbin),np.max(dbin),100)
bestfit5 = wpfunc5(R0best5)	# Gamma = 1.5
bestfit8 = wpfunc8(R0best8)	# Gamma = 1.8

fig,ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(fitline*1000, bestfit8(fitline), c='orange')		# Gamma = 1.8
ax.plot(fitline*1000, bestfit5(fitline), c='gold', ls='--')	# Gamma = 1.5
ax.errorbar(dbin_cent*1000, wGG, yerr=wGG_sigma, mec='orange', mfc='gold', ecolor='orange', fmt='o', ms=8, capsize=2)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: r'${:g}$'.format(y))) 
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: r'${:g}$'.format(y))) 
ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: r'${:g}$'.format(y)))
ax.set_xlabel(r'$\rm R~{\rm(h^{-1}~\mathrm{ckpc})}$')
ax.set_xticks([50,100,200,300,400,500,600]) 
ax.set_ylabel(r'$\rm w_p~{\mathrm{(R)}}$')
ax.legend(loc='best')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(45,600)

plt.show()






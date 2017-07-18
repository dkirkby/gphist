import numpy as np
import matplotlib.pyplot as plt


c_of_light = 299792.458 #km/s
r_d = 147.36 #Mpc

z_H0 = 0.0
H0 = 73.24 #km /s /Mpc
sigma_H0 = 1.74 #km /s /Mpc

z_Lya = 2.3
DH_Lya = 9.15*r_d
H_Lya = c_of_light / DH_Lya
sigma_H_Lya = H_Lya *(0.20/9.15)



BOSS_data = np.loadtxt('BAO_BEUTLER_results_nostrings.txt')
BOSS_cov_z1 = np.loadtxt('BAO_BEUTLER_cov_z1.txt')
BOSS_cov_z2 = np.loadtxt('BAO_BEUTLER_cov_z2.txt')
BOSS_cov_z3 = np.loadtxt('BAO_BEUTLER_cov_z3.txt')

z_BOSS = BOSS_data[:,0]
H_BOSS = BOSS_data[:,1]

print H_BOSS

sigma_H_BOSS = np.sqrt(np.array([BOSS_cov_z1[0,0],BOSS_cov_z2[0,0],BOSS_cov_z3[0,0]]))

print sigma_H_BOSS





plt.errorbar(z_H0, H0/(1+z_H0), yerr=sigma_H0/(1+z_H0), fmt='ro', label=r'H$_0$')
plt.errorbar(z_Lya, H_Lya/(1+z_Lya), yerr=sigma_H_Lya/(1+z_Lya), fmt='g^', label=r'Ly$\alpha$')
plt.errorbar(z_BOSS, H_BOSS/(1+z_BOSS), yerr=sigma_H_BOSS/(1+z_BOSS), fmt='bs', label='BOSS galaxies')
plt.grid(True)
plt.xlim(0,2.5)
plt.ylim(56,72)
plt.xlabel(r'redshift, $z$')
plt.ylabel(r'$H(z)/(1+z) \times r_{d147}$ [km s$^{-1}$ Mpc$^-1$]')
plt.legend(loc='best')
plt.savefig('consistency_Bautista17.png')
plt.clf()
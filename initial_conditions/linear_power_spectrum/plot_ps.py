import numpy as np
import matplotlib.pyplot as plt

#plt.loglog()
#x,y=np.loadtxt('extended_planck_linear_powspec', unpack=True)
#
#plt.plot(x,y, label='extended_planck_linear_powspec')
#
#x,y=np.loadtxt('Eagle_2013_matterpower.dat', unpack=True)
#
#plt.plot(np.log10(x),np.log10(x**3*y), label='Eagle_2013_matterpower', ls='--')
#
#x,y=np.loadtxt('planck_march_2013_matterpower.dat', unpack=True)
#
#plt.plot(np.log10(x),np.log10(x**3*y), label='planck_march_2013_matterpower', ls='--')

#x1,y1=np.loadtxt('EAGLE_XL_powspec_18-07-2019.txt', unpack=True)
#plt.plot(x1,y1, label='EAGLE_XL_powspec_18-07-2019.txt')

f, axarr = plt.subplots(2)

#x,y=np.loadtxt('ic_gen_power_spectra/EAGLE_XL_powspec_18-07-2019.txt', unpack=True)
#axarr[0].plot(x,y, label='eagle-xl', ls='--')

x2,y2=np.loadtxt('raw_power_spectra/new_z0', unpack=True)
axarr[0].plot(x2,y2, label='z0', ls='--')

for this_z in ['2']:
    x3,y3=np.loadtxt('raw_power_spectra/new_z%s'%this_z, unpack=True)
    axarr[0].plot(x3,y3, label='z%s'%this_z, ls='--')

    axarr[1].plot(x2, np.true_divide(y2,y3))

#x2,y2=np.loadtxt('stu', unpack=True)
#y2 = np.log10(x2**3*y2)
#x2=np.log10(x2)
#plt.plot(x2,y2, label='stu', ls='--')

#x1,y1=np.loadtxt('extended_planck2018_linear_powspec', unpack=True)
#plt.plot(x1,y1, label='extended_planck2018_linear_powspec', ls='dotted')

#axarr[1].plot(x1, np.true_divide(x1,x2))

#plt.xlim(-2,2)
axarr[0].legend()

#axarr[1].set_ylim(0.97,1.03)
axarr[0].set_yscale('log')
axarr[0].set_xscale('log')
axarr[1].set_xscale('log')
plt.show()
#plt.savefig('power_spectrum_plot.pdf')
#plt.close()

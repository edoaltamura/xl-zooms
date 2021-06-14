import sys, os
import numpy as np
from matplotlib import pyplot as plt
from unyt import Solar_Mass

sys.path.append("..")
from literature import Sun2009, Pratt2010

plt.style.use('../register/mnras.mplstyle')

fig, axes = plt.subplots()

axes.set_xscale('log')
axes.set_yscale('log')
axes.axvline(0.15, color='k', linestyle='--', lw=0.5, zorder=0)
axes.set_ylabel(r'$K/K_{500}$')
axes.set_xlabel(r'$r/r_{500}$')
axes.set_ylim([5e-3, 20])
axes.set_xlim([5e-3, 2.5])

sun_observations = Sun2009()
r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

axes.fill_between(
    r_r500,
    S_S500_10,
    S_S500_90,
    color='grey', alpha=0.4, linewidth=0
)
axes.plot(r_r500, S_S500_50, c='grey', label=sun_observations.citation)

rexcess = Pratt2010()
bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
    m500_limits=(
        1e14 * Solar_Mass,
        5e14 * Solar_Mass
    ),
    k500_rescale=True
)

axes.fill_between(
    rexcess.radial_bins,
    bin_perc16,
    bin_perc84,
    color='aqua',
    alpha=0.4,
    linewidth=0
)
axes.plot(rexcess.radial_bins, bin_median, c='aqua', label=rexcess.citation)
axes.plot(np.array([0.01, 2.5]), 1.40 * np.array([0.01, 2.5]) ** 1.1, c='grey', ls='--', label='VKB (2005)')

r = [0.01056767, 0.01180153, 0.01317944, 0.01471824, 0.0164367,
     0.0183558, 0.02049898, 0.02289238, 0.02556524, 0.02855016,
     0.0318836, 0.03560625, 0.03976354, 0.04440622, 0.04959097,
     0.05538108, 0.06184722, 0.06906834, 0.07713257, 0.08613836,
     0.09619564, 0.10742719, 0.1199701, 0.13397748, 0.14962033,
     0.16708959, 0.18659852, 0.20838526, 0.23271576, 0.25988703,
     0.29023074, 0.32411729, 0.36196036, 0.40422187, 0.45141772,
     0.50412404, 0.5629842, 0.62871671, 0.70212398, 0.78410208,
     0.87565172, 0.97789046, 1.0920663, 1.21957302, 1.36196707,
     1.52098667, 1.69857297, 1.89689376, 2.11836996, 2.36570511]
k = [0.08939827, 0.09195352, 0.09570614, 0.09726039, 0.10143045,
     0.1054346, 0.10805542, 0.11289545, 0.11614708, 0.11789873,
     0.12076289, 0.12354064, 0.12707706, 0.1277981, 0.12853718,
     0.12916432, 0.13103351, 0.13395326, 0.13809182, 0.14351657,
     0.1497746, 0.1669413, 0.17915367, 0.18797427, 0.1975279,
     0.20990694, 0.22427633, 0.24233486, 0.26965544, 0.29354015,
     0.32372862, 0.35085532, 0.37696299, 0.41013053, 0.44491786,
     0.48157999, 0.53423452, 0.59310865, 0.64431387, 0.6772294,
     0.82715368, 1.09084082, 1.264382, 1.28105307, 1.34895051,
     1.62680733, 1.99230754, 2.24905324, 2.54650331, 2.72191]
axes.plot(r, k, label='VR18 adiabatic')

k = [0.1593734, 0.18088278, 0.2228837, 0.24383876, 0.24651441,
     0.31293085, 0.34132445, 0.35963014, 0.39034978, 0.42454255,
     0.43724528, 0.44203937, 0.44408238, 0.46002173, 0.45804211,
     0.46055579, 0.46739706, 0.47504455, 0.47832716, 0.48750401,
     0.49540627, 0.50917357, 0.52394694, 0.53522253, 0.55067492,
     0.57413417, 0.59835428, 0.6178053, 0.64076531, 0.66509688,
     0.68879676, 0.71568954, 0.7492156, 0.77882963, 0.80158263,
     0.82287711, 0.85083896, 0.90637338, 0.95664042, 1.01831222,
     1.11262858, 1.24661756, 1.37374949, 1.50637972, 1.67239177,
     1.78120422, 1.94763279, 2.19598317, 2.2483139, 2.19000483]
axes.plot(r, k, label='VR18 AGNdT8.5')

k = [0.17063178, 0.18462509, 0.15067273, 0.1642565, 0.18069917,
     0.20468523, 0.20258166, 0.25256845, 0.29330206, 0.3291648,
     0.36726138, 0.39712864, 0.42348215, 0.44717211, 0.46150696,
     0.46381617, 0.46969348, 0.48450655, 0.491772, 0.50034457,
     0.50581348, 0.51010036, 0.51122856, 0.51747596, 0.52223909,
     0.53313881, 0.54496443, 0.55688685, 0.57226402, 0.59490997,
     0.61641407, 0.64280665, 0.67149442, 0.70194876, 0.73707378,
     0.780653, 0.81962609, 0.87521487, 0.93723601, 1.02184606,
     1.11755681, 1.25651538, 1.38538027, 1.53131187, 1.6866796,
     1.8360703, 2.02452016, 2.21023726, 2.33550382, 2.40314436]
axes.plot(r, k, label='VR18 AGNdT7.5')

k = [0.34919494, 0.43282777, 0.28952217, 0.3592802, 0.40497199,
     0.55060828, 0.51199633, 0.61840367, 0.54981875, 0.64775234,
     0.62897629, 0.69748622, 0.75461, 0.74047154, 0.80728155,
     0.85164976, 0.86189187, 0.86025214, 0.86399853, 0.87910783,
     0.89611399, 0.90446275, 0.91774428, 0.92549551, 0.93270123,
     0.93763059, 0.94241309, 0.94536591, 0.95412511, 0.96590018,
     0.98421305, 0.99893653, 1.00175822, 1.0048207, 1.0205369,
     1.05205107, 1.07446504, 1.09381604, 1.1185652, 1.16407275,
     1.24216425, 1.33619773, 1.42108917, 1.51808202, 1.63339543,
     1.84410977, 2.04335594, 2.17920423, 2.2351532, 2.20658994]
axes.plot(r, k, label='VR18 AGNdT9')

plt.legend()
plt.show()
plt.close()

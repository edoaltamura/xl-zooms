Eagle-XL Zooms
==========
![EAGLE-XL Logo](./xl-logo-light-crop.png)

EAGLE-XL is the next generation of cosmological simulations of periodic cubic volumes, created with the [SWIFT](https://github.com/SWIFTSIM) 
hydrodynamical code and its derivatives: [Velociraptor](https://github.com/SWIFTSIM/velociraptor-python) for halo finding 
and [Swiftsimio](https://github.com/SWIFTSIM/swiftsimio) for handling initial conditions and reading outputs.

Zoom simulations
------------
Zoom-in simulations are high-resolution runs of a given cosmological volume, where the particle resolution is highly
enhanced in correspondence of objects of interests (typically clusters or galaxy groups) and lowered elsewhere.
The zoomed halos are chosen to be isolated objects in one of the EAGLE-XL boxes at present redshift. Three halos
are currently selected within the mass range   10<sup>13</sup> &leq; M<sub>200</sub>/M<sub>&odot;</sub> &leq; 10<sup>14</sup>.

Computing architecture
------------
Zooms are run with a single-node configuration, preferably on the `cosma7` computer cluster, equipped with 
452 compute nodes, each with 512GB RAM and 28 cores (2x Intel Xeon Gold 5120 CPU @ 2.20GHz). Visit the [COSMA-DiRAC pages](https://www.dur.ac.uk/icc/cosma/)
for more info.

An important task of this project is to assess the possibility to run high-resolution zooms with SWIFT on one single node,
first with dark matter only, then with baryonic physics integrated.
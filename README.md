Eagle-XL Zoom Simulations
==========
![EAGLE-XL Logo](img/xl-logo-light-crop.png)

EAGLE-XL is the next generation of cosmological simulations of periodic cubic volumes, created with the [SWIFT](https://github.com/SWIFTSIM) 
hydrodynamical code and its derivatives: [Velociraptor](https://github.com/SWIFTSIM/velociraptor-python) for halo finding 
and [Swiftsimio](https://github.com/SWIFTSIM/swiftsimio) for handling initial conditions and reading outputs.

Why zoom simulations?
------------
Zoom-in simulations are high-resolution runs of a given cosmological volume, where the particle resolution is highly
enhanced in correspondence of objects of interests (typically clusters or galaxy groups) and lowered elsewhere.
The zoomed halos are chosen to be isolated objects in one of the EAGLE-XL boxes at present redshift. Three halos
are currently selected within the mass range   10<sup>13</sup> &leq; M<sub>200</sub>/M<sub>&odot;</sub> &leq; 10<sup>14</sup>.

Computing architecture
------------
Zooms are run with a single-node configuration and preferably on the `cosma7` computer cluster. The table below summarises
the properties of the computing clusters available to the Virgo Consortium. Visit the [COSMA-DiRAC pages](https://www.dur.ac.uk/icc/cosma/)
for more info. 

| Machine name       | Memory/node   | Cores/node  | CPU info                                       | Total nodes        |
| ------------------ |:-------------:|:-----------:|:----------------------------------------------:|:------------------:|
| Cosma 5            | 128 GB        |   16        |   2x Intel Xeon CPU E5-2670 0 @ 2.60GHz        |   675              |
| Cosma 6            | 128 GB        |   16        |   2x Intel Xeon CPU E5-2670 0 @ 2.60GHz        |   675              |
| Cosma 7            | 512 GB        |   28        |   2x Intel Xeon Gold 5120 @ 2.20GHz        |   452              |
| Cosma 8 (compute)  | 1 TB          |   64        |   2x AMD EPYC 7H12 water-cooled @ 2.6GHz   |   32               |
| Cosma 8 (monolith) | 4 TB          |   64        |   2x AMD EPYC 7702 @ 2.6GHz                    |   1                |

An important task of this project is to assess the possibility to run high-resolution zooms with SWIFT on one single node,
first with dark matter only, then with baryonic physics integrated. Due to the intensive use of memory, the final version
of the hydro simulations might make use of the `cosma8`'s large capacity nodes. 

Zoom pipeline
------------

The parent box
------------

Initial conditions
------------

Redshift evolution
------------

Analysis tools
------------

<sup>Logo design by Edo Altamura &copy; 2020</sup>
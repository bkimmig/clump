# clumpy

clumPy (CLUster Membership Python) a tool for determining kinematic members of a globular cluster.

Using radial velocity, and the distance from the center of the cluster, we can assign membership probabilities to all of the stars. This uses Expectation maximization to determine the membership probabilities. The assumed distribution of the velocities of the clusters are Gaussian. If applicable, you can include a second parameter (e.g. metallicity, log g) to aid in the fit. Finally, you can also include a light profile, like a King (1966) model to aid in the fit.

This is the primary cluster membership code used in the following papers: 

<a href='http://adsabs.harvard.edu/abs/2015AJ....149...53K'> Measuring Consistent Masses for 25 Milky Way Globular Clusters </a>

<a href='http://arxiv.org/abs/1509.06391'> Evidence That Hydra I is a Tidally Disrupting Milky Way Dwarf Galaxy </a>

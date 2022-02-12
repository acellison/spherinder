# spherinder
Stretched sphere spectral methods for efficient rotating fluid dynamics simulations

Spherinder depends on a dedalus and dedalus-sphere installation.  To install dedalus,
follow instructions in the dedalus-project documentation:
    https://dedalus-project.readthedocs.io/en/latest/pages/installation.html
If using conda (recommended), activate the dedalus environemnt.
Next download the source for dedalus sphere and follow its installation instructions:
    https://github.com/DedalusProject/dedalus_sphere

Our preferred sparse eigensolver for Spherinder problems is the scikits umfpack solver.
In the conda environment run "conda install -c conda-forge scikit-umfpack" to install it.

To install the spherinder python package, run the command "pip install -e ."
This will create an editable installation of the scripts inside "spherinder"
in an importable module called "spherinder".

The scripts for generating figures from the paper are in the paper/ directory.
These are a good place to start for example usage of the spherinder computational
basis.

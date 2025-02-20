# QuickQuake
*Simple and streamlined workflow for earthquake detection and relocation*

![plot1](/figs/QuickQuakeLogo.png)

This repository contains the Python scripts for wrapping up and streamlining the procedures and computer codes for earthquake detection and relocation. Some scripts are modified from QuakeFlow. For detailed description and usage of each individual package, please read the corresponding documentation.

## Major earthquake detection packages used in this workflow.
* QuakeFlow: workflow that wraps multiple detection and association methods.
* PhaseNet: a package for earthquake phase picks (detection)
* GAMMA: phase associator.
* HypoInverse: earthquake relocation program (in Fortran).
* HypoInvPy: Python interface for running Hypoinverse.

## Package structure
* quickquake: contains the main functions/modules that can be called from any working directory.
* examples: contains example scripts/jupyter notebooks to run this package.
* figs: figures for the README.md and other examples.

## Credits
* This workflow is developed by Elizabeth Alzate Gutierrez & Xiaotao Yang.
* Scripts that were modified from other packages are clearly commented at the beginning of the codes.
* Please cite corresponding references for each individual package.
  



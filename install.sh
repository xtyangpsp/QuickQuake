#!/bin/bash
profilename='.bash_profile'
qqdir='quickquake'
# Things to include in this file
# 1. Procedures to install individual packages

# 2. Make the scripts execuatable 
cd $qqdir
for f in `ls *.py`
do
	chmod a+x $f
done
cd ..
# 3. Add the quickquake path to system search path within .bashrc or .bash_profile
rootdir=`pwd`
echo $rootdir
echo export PATH=${rootdir}/${qqdir}:"\$PATH" >> ~/${profilename}

# 4. Refresh profile
echo Please run source "~/${profilename}" to apply the changes to the profile.
#source ~/${profilename}

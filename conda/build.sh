#! /bin/bash -e 

# before running anaconda should be installed on computer


rm -rf stream-learn/

# create temporary environment (current user env can cause conflits with packages we want to install)
conda create -y -n tmp-conda-upload python=3.8
eval "$(conda shell.bash hook)"
conda activate tmp-conda-upload
conda install -y conda-build anaconda-client grayskull -c conda-forge

# build conda package from the latest pypi release
grayskull pypi stream-learn 
conda-build stream-learn -c conda-forge -c w4k2

# upload package to anaconda
echo "login to anaconda"
anaconda whoami &> /tmp/conda_user.txt
CURRENT_ANACONDA_USER=`cat /tmp/conda_user.txt | cut -d$'\n' -f2`
if [ "$CURRENT_ANACONDA_USER" = "Anonymous User" ]; then
    anaconda login
fi
CONDA_INSTALLATION_DIR=`which conda | xargs dirname | xargs dirname`
LATEST_PACKAGE=`ls -t $CONDA_INSTALLATION_DIR/conda-bld/noarch/stream-learn-* | head -1`
anaconda upload -u w4k2 $LATEST_PACKAGE

# cleanup
conda deactivate
conda env remove -n tmp-conda-upload
#! /bin/bash -e 

# before running anaconda should be installed on computer


rm -rf stream-learn/

# create temporary environment (current user env can cause conflits with packages we want to install)
conda create -y -n tmp-stream-learn-upload python=3.8
source activate tmp-stream-learn-upload
conda install -y conda-build anaconda-client grayskull -c conda-forge

# build conda package from the latest pypi release
grayskull pypi stream-learn 
conda-build stream-learn -c conda-forge

# upload package to anaconda
echo "login to anaconda"
anaconda login
CONDA_INSTALLATION_DIR=`which conda | xargs dirname | xargs dirname`
LATEST_PACKAGE=`ls -t $CONDA_INSTALLATION_DIR/conda-bld/linux-64/stream-learn-* | head -1`
anaconda upload -u w4k2 $LATEST_PACKAGE

# cleanup
conda deactivate
conda env remove -n tmp-stream-learn-upload
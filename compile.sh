#!/bin/bash
setup_env=${1}

if [ "$setup_env" = "True" ]; then
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install --upgrade pip
    pip install editdistance
    pip install Polygon3
    pip install pyclipper
    pip install Cython
    pip install openmim
    mim install mmcv==1.7.2
    pip install prefetch_generator
    pip install scipy
    pip install yacs
    pip install tqdm
    pip install opencv-python==4.6.0.66
fi

cd ./models/post_processing/pa/
python setup.py build_ext --inplace
cd ../../../

cd ./models/post_processing/pse/
python setup.py build_ext --inplace
cd ../../../

cd ./models/post_processing/ccl/
python setup.py build_ext --inplace
cd ../../../


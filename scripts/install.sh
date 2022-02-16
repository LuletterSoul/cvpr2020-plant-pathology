conda create -n fgcv python=3.7
conda activate -n fgcv
pip install -r requriement.txt  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
conda install -c conda-forge nvidia-apex
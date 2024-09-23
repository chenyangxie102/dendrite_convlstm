#!/bin/bash
#SBATCH -J torch1.10
#SBATCH -p wzhdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4 

#module unload compiler/dtk/22.04.2
module rm compiler/dtk/21.10
module load compiler/dtk/23.10
source /work/home/maohong/miniconda/etc/profile.d/conda.sh
conda activate yangpytorch_dtk

#export LD_LIBRARY_PATH=~/miniconda/envs/yangpytorch_dtk/lib:$LD_LIBRARY_PATH

python3 cikm_convlstm_run.py   # xxx.py 改为您的输入文件名#

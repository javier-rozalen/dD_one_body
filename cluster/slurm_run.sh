#!/bin/bash
#SBATCH -N 1 # number of nodes i
#SBATCH -n 1 # number of cores
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=javirozalen.code@gmail.com

echo $1
python3 3d_HO.py $1

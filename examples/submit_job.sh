#!/bin/bash

#SBATCH --job-name=ts_searches
#SBATCH --partition=ql40s
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

source /comm/swstack/bin/modules.sh --force
source ~/.bashrc
conda activate ocp

python run_calculations_ocp_mechanism.py >> output.txt

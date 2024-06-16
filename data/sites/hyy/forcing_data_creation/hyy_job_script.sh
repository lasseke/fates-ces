#!/bin/bash
#SBATCH --account=nn2806k
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --job-name=HYY-data
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --time=12:00:00

set -o errexit  # Exit the script on any error

module --quiet purge  # Reset the modules to the system default
module load git/2.36.0-GCCcore-11.3.0-nodocs
module load Anaconda3/2022.05

eval "$(/cluster/software/Anaconda3/2022.05/bin/conda shell.bash hook)"
source activate /cluster/work/users/lassetk/conda_envs/subset-data-env

python3 create_data.py \
    --ctsm-root /cluster/work/users/lassetk/CTSM \
    --cesm-data-root /cluster/shared/noresm/inputdata \
    --output-dir /cluster/shared/noresm/sites \
    --sites hyy_site_info.json \
    --cpu-count 10

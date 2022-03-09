#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name="hidden_vision"
#SBATCH --output=slurm-%A-%x.out
#SBATCH --account="punim0784"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time 168:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Run the job from this directory:
cd /data/gpfs/projects/punim0784/hanxunh/hidden_vision

# The modules to load:
source /usr/local/module/spartan_new.sh
module load web_proxy
module load gcc/8.3.0
module load cuda/10.1.243
module load openmpi/3.1.4
module load python/3.8.2

nvidia-smi

exp_config=$1
exp_path=$2
exp_name=$3
seed=$4

echo 'exp_path: ' $exp_path
echo 'exp_config: ' $exp_config
echo 'exp_name: ' $exp_name

# Train
python train.py --exp_path $exp_path \
                --exp_config $exp_config \
                --exp_name $exp_name \
                --seed $seed

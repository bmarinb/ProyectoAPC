#!/bin/bash
#SBATCH --workdir=/home/username                            # cd to path
#SBATCH --ntasks=1                                          # run 1 task
#SBATCH --nodelist=scylla                                   # list of nodes to be used: scylla
#SBATCH --mail-user=user@uc.cl                              # send email to user@uc.cl when finished
#SBATCH --output=%A-%a.log                                  # save logs to %A-%a.log
#SBATCH --mail-type=ALL                                     # enable email confirmation
#SBATCH --cpus=4                                            # request 4 CPUs to slurm for each task
#SBATCH --gres=gpu:1                                        # request 1 gpus to slurm
#SBATCH --partition=ialab-high                              # request access to ialab-high partition
#SRUN --workdir=/home/username                              # cd to path

pwd; hostname; date
source new_env/bin/activate                                     # activate virtualenv
echo "Begin!"
python main.py


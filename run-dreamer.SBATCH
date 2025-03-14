#!/bin/bash

#SBATCH --output=/scratch/ss19021/dreamerv3-mod/slurm_outputs/%x_%j.out
#SBATCH --error=/scratch/ss19021/dreamerv3-mod/slurm_outputs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ss19021@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:59:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=dreamer

module purge

singularity exec --nv --overlay /scratch/ss19021/dreamerv3-mod/dreamerv3-mod.ext3:rw /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; cd /home/ss19021/dreamerv3-mod/; bash run_main.sh"

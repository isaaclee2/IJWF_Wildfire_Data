#!/bin/bash
#SBATCH --account=aoberai_286
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --output=output_%x-%J.out
#SBATCH --error=error_%x-%J.out
#SBATCH --job-name=fire_area_prediction
#SBATCH --mail-user=ihlee@usc.edu
#SBATCH --mail-type=ALL

module purge
eval "$(conda shell.bash hook)"
conda activate diffusion_env
python main.py --config config.yml --save_dir ./run_wildfire --train --ni
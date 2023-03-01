#!/bin/bash
#SBATCH --job-name=dViT100k
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=biggpu
#SBATCH --output=/home-mscluster/erex/research_project/models/dino/experiments/pre-trainingViT100k/result-%A-%a.out
#SBATCH --array=1-1 # job array index


echo "running job: "${SLURM_ARRAY_TASK_ID}
mkdir -p /home-mscluster/erex/research_project/models/dino/experiments/pre-trainingViT100k/saving_dir${SLURM_ARRAY_TASK_ID}

python3 -m torch.distributed.run --nproc_per_node=2 main_dino.py \
 --data_path /home-mscluster/erex/research_project/datasets/NLST_100k \
 --output_dir /home-mscluster/erex/research_project/models/dino/experiments/pre-trainingViT100k/saving_dir${SLURM_ARRAY_TASK_ID} \
 --epochs 500 \
 --arch vit_small \

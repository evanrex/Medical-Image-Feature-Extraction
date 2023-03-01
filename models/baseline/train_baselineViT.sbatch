#!/bin/bash
#SBATCH --job-name=baseViT
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=bigbatch
#SBATCH --output=/home-mscluster/erex/research_project/models/baseline/experiments/trainingViT/result-%A-%a.out
#SBATCH --array=1-5 # job array index


echo "running job: "${SLURM_ARRAY_TASK_ID}
mkdir -p /home-mscluster/erex/research_project/models/baseline/experiments/trainingViT/saving_dir${SLURM_ARRAY_TASK_ID}

python3 -m torch.distributed.run --nproc_per_node=1 eval_linear.py \
--pretrained_weights /home-mscluster/erex/research_project/models/baseline/experiments/pre-trainingViT/saving_dir1/checkpoint.pth \
--data_path /home-mscluster/erex/research_project/datasets/Covidx-CT \
--output_dir /home-mscluster/erex/research_project/models/baseline/experiments/trainingViT/saving_dir${SLURM_ARRAY_TASK_ID} \
--arch vit_small \
--num_labels 2

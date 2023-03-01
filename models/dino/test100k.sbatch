#!/bin/bash
#SBATCH --job-name=d100ktest
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/erex/research_project/models/dino/experiments/training100k/test_result-%A-%a.out
#SBATCH --array=1-3 # job array index


echo "running job: "${SLURM_ARRAY_TASK_ID}

python3 -m torch.distributed.run --nproc_per_node=1 eval_linear.py \
--pretrained_weights /home-mscluster/erex/research_project/models/dino/experiments/pre-training100k/saving_dir${SLURM_ARRAY_TASK_ID}/checkpoint.pth \
--pretrained_linear_weights /home-mscluster/erex/research_project/models/dino/experiments/training100k/saving_dir${SLURM_ARRAY_TASK_ID}/checkpoint.pth.tar \
--data_path /home-mscluster/erex/research_project/datasets/Covidx-CT \
--output_dir /home-mscluster/erex/research_project/models/dino/experiments/training100k/saving_dir${SLURM_ARRAY_TASK_ID} \
--arch resnet50 \
--num_labels 2 \
--evaluate_on_test


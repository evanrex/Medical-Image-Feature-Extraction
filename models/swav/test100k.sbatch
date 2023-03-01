#!/bin/bash
#SBATCH --job-name=swavtrain
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/erex/research_project/models/swav/experiments/training100k/test_result-%A-%a.out
#SBATCH --array=1-3 # job array index


echo "running job: "${SLURM_ARRAY_TASK_ID}
mkdir -p /home-mscluster/erex/research_project/models/swav/experiments/training100k/saving_dir${SLURM_ARRAY_TASK_ID}

python3 -m torch.distributed.run --nproc_per_node=1 eval_linear.py \
--pretrained /home-mscluster/erex/research_project/models/swav/experiments/pre-training100k/saving_dir${SLURM_ARRAY_TASK_ID}/checkpoint.pth.tar \
--pretrained_linear_weights /home-mscluster/erex/research_project/models/swav/experiments/training100k/saving_dir${SLURM_ARRAY_TASK_ID}/checkpoint.pth.tar \
--data_path /home-mscluster/erex/research_project/datasets/Covidx-CT \
--dump_path /home-mscluster/erex/research_project/models/swav/experiments/training100k/saving_dir${SLURM_ARRAY_TASK_ID} \
--arch resnet50 \
--num_labels 2 \
--evaluate_on_test


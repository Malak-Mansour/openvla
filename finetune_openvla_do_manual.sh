#!/bin/bash
#SBATCH --job-name=openvla-finetune
#SBATCH --output=logs/openvla-finetune-%j.out
#SBATCH --error=logs/openvla-finetune-%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate openvla
cd ~/Downloads/ICL/openvla


torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /l/users/malak.mansour/Datasets/do_manual/rlds \
  --dataset_name do_manual \
  --run_root_dir /l/users/malak.mansour/OpenVLA/runs/do_manual \
  --adapter_tmp_dir /l/users/malak.mansour/OpenVLA/adapters/tmp_do_manual \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 500 \
  --wandb_entity mim7995-mbzuai \
  --wandb_project openvla-do_manual

# launch with sbatch finetune_openvla_libero.sh
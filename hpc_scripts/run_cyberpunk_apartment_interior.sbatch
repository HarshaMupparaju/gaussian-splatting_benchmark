#!/bin/bash
##SBATCH --job-name=gaussian_splatting
#SBATCH --output=/scratch/sm12754/projects/gaussian_splatting/logs/%A.out
#SBATCH --error=/scratch/sm12754/projects/gaussian_splatting/logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=harshadiablo@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000|a100|h100
#SBATCH --account=pr_173_tandon_advanced

RUN_DIR=$HOME/gaussian-splatting_benchmark;

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif;

OVERLAY_FILE=$SCRATCH/gaussian_splatting/overlay-25GB-500K.ext3;

DATASET_DIR=$SCRATCH/gaussian_splatting/data/cyberpunk_apartment_interior;

singularity exec --nv --overlay $OVERLAY_FILE:ro $SINGULARITY_IMAGE /bin/bash -c "
    source /ext3/env.sh
    conda activate gaussian_splatting
    cd $RUN_DIR &&
    python train.py -s $DATASET_DIR -r 1 \
                    --model_path $DATASET_DIR/output \
                    -d $DATASET_DIR/depths \
                    --port 5001 \
                    --checkpoint_iterations 30000 \
                    --eval
"
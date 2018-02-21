#!/bin/bash

#SBATCH --array=1-5
#SBATCH -p serial
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=48:00:00

module purge
module load anaconda/2-4.1.1
source activate tf


### Customize these lines ###

gold="QALB.dev"  # make sure to drop the .gold or .m2 extension
case "$SLURM_ARRAY_TASK_ID" in
"1") dataset="QALB.dev.orig" config="orig --extension=orig" ;;
"2") dataset="QALB.dev.mle" config="mle --extension=mle" ;;
"3") dataset="QALB.dev.mada" config="mada --extension=mada" ;;
"4") dataset="QALB.dev.mada.mle" config="mada_mle --extension=mada.mle" ;;
"5") dataset="QALB.dev.mada.mle" config="mle_mada --extension=mle.mada" ;;
esac

### END ###


### Training ###
python3 -m ai.tests.qalb --model_name=$config

### Evaluations ###
model_name=$(echo $config | cut -d' ' -f 1)

# Decoder
python3 -m ai.tests.qalb --decode=ai/datasets/data/qalb/$dataset --output_path=output/$model_name/decoder.out --model_name=$config

# M2 scorer
python2 ai/tests/m2scripts/m2scorer.py -v --beta 1 output/$model_name/decoder.out ai/datasets/data/qalb/$gold.m2 > output/$model_name/m2scorer.out

# Full evaluations + readable output
python3 analysis.py output/$model_name/m2scorer.out > output/$model_name/analysis.out

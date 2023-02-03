#!/bin/bash
subscript=slurm_run.sh
for ((i=0 ; i<8 ; i++)) ; do
  jobname=3D_HO${i}
  echo $jobname
  sbatch --job-name=$jobname --priority=TOP $subscript $i
  #$subscript $i
done

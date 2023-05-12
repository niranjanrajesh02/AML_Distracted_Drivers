#! /bin/bash
#PBS -N augtestvanilla
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=90
#PBS -q gpu

rm -r /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/aug_test/Results
mkdir /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/aug_test/Results
rm /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/aug_test/out.log
rm /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/aug_test/err.log
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate computer-vision
python /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/aug_test/eval.py



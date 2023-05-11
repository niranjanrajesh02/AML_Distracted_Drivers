#! /bin/bash
#PBS -N vit
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=50
#PBS -q gpu

rm /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/ViT/out.log
rm /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/ViT/err.log
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate computer-vision
python /home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/ViT/cifar_classify.py


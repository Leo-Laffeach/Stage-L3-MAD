#!/usr/bin/env bash

setcuda 11.3

source torch_env/bin/activate

export OPENBLAS_NUM_THREADS=2


nohup $(which python)  tests_mean.py > dossier_sauvegarde_mean.out &

nohup $(which python)  tests_OT_Muzellec.py > dossier_sauvegarde_OT_Muzellec.out &

nohup $(which python)  tests_OT_MAD.py > dossier_sauvegarde_OT_MAD.out &

nohup $(which python)  tests_OT_Muzellec_MAD.py > dossier_sauvegarde_OT_Muzellec_MAD.out &

nohup $(which python)  tests_RR_Muzellec_linear.py > dossier_sauvegarde_RR_Muzellec_linear.out &

nohup $(which python)  tests_RR_MAD_linear.py > dossier_sauvegarde_RR_MAD_linear.out &

nohup $(which python)  tests_RR_Muzellec_mlp.py > dossier_sauvegarde_RR_Muzellec_mlp.out &

nohup $(which python)  tests_RR_MAD_mlp.py > dossier_sauvegarde_RR_MAD_mlp.out &
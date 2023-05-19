#!/bin/bash

PS3='Please enter your choice: '
options=("Run Galactica"
         "Run LLaMA"
         "Run RST"
         "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Run Galactica")
            echo "you chose Run Galactica."

            export MODEL_NAME=Galactica
            export MODEL_TYPE=standard
            export DATA_REPO_PATH=/lambda_stor/homes/ac.gpark/BioIE-LLM/data
            export OUTPUT_DIR=/lambda_stor/homes/ac.gpark/BioIE-LLM/result
            export DATA_NAME=string
            export TASK=entity
            export BATCH_SIZE=8
            export N_SHOTS=2

            python ~/BioIE-LLM/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --batch_size $BATCH_SIZE \
                --n_shots $N_SHOTS \
                --parallelizm
            
            : '
                --parallelizm
            '
            
            break
            ;;
        "Run LLaMA")
            echo "you chose Run LLaMA."
            
            export MODEL_NAME=LLaMA
            export MODEL_TYPE=/scratch/ac.gpark/LLaMA_HF/7B
            export DATA_REPO_PATH=/lambda_stor/homes/ac.gpark/BioIE-LLM/data
            export OUTPUT_DIR=/lambda_stor/homes/ac.gpark/BioIE-LLM/result
            export DATA_NAME=string
            export TASK=entity
            export BATCH_SIZE=4
            export N_SHOTS=2

            python ~/BioIE-LLM/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --batch_size $BATCH_SIZE \
                --n_shots $N_SHOTS
                
            break
            ;;
        "Run RST")
            echo "you chose Run RST."
            
            export MODEL_NAME=RST
            export MODEL_TYPE=XLab/rst-all-11b
            export DATA_REPO_PATH=/lambda_stor/homes/ac.gpark/BioIE-LLM/data
            export OUTPUT_DIR=/lambda_stor/homes/ac.gpark/BioIE-LLM/result
            export DATA_NAME=string
            export TASK=entity
            export BATCH_SIZE=4
            export N_SHOTS=2

            python ~/BioIE-LLM/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --batch_size $BATCH_SIZE \
                --n_shots $N_SHOTS
                
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
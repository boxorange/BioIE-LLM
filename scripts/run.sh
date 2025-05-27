#!/bin/bash

PS3='Please enter your choice: '
options=("Run Galactica"
         "Run LLaMA"
         "Run Alpaca"
         "Run LLaMA-2"
         "Run LLaMA-3"
         "Run LLaMA-3.1"
         "Run Mistral"
         "Run Solar"
         "Run Falcon"
         "Run MPT"
         "Run RST"
         "Run BioGPT"
         "Run BioMedLM"
         "Run Gemini"
         "Quit")
select opt in "${options[@]}"
do
    export DATA_REPO_PATH=/home/ac.gpark/BioIE-LLM-WIP/data
    export OUTPUT_DIR=/home/ac.gpark/BioIE-LLM-WIP/result
    export LORA_OUTPUT_DIR=/scratch/ac.gpark/LoRA_finetuned_models
    
    case $opt in
        "Run Galactica")
            echo "you chose Run Galactica."

            export MODEL_NAME=Galactica
            export MODEL_TYPE=facebook/galactica-6.7b
            # export MODEL_TYPE=facebook/galactica-30B
            # export MODEL_TYPE=facebook/galactica-120b
            export DATA_NAME=kbase # string, kegg, indra, kbase, lll
            export TASK=entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-2-7b-chat-hf/string/entity/final_checkpoint
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                
            
            : '
                --parallelizm 
            '
            
            break
            ;;
        "Run LLaMA")
            echo "you chose Run LLaMA."
            
            export MODEL_NAME=LLaMA
            export MODEL_TYPE=/scratch/ac.gpark/LLaMA_HF/7B
            export DATA_NAME=string # string, kegg, indra, kbase
            export TASK=entity # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase)
            export TEST_BATCH_SIZE=32
            export TRAIN_BATCH_SIZE=32 # used in finetuning
            export N_SHOTS=1

            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --test_batch_size $TEST_BATCH_SIZE \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --n_shots $N_SHOTS
                
            break
            ;;
        "Run Alpaca")
            echo "you chose Run Alpaca."
            
            export MODEL_NAME=Alpaca
            export MODEL_TYPE=/scratch/ac.gpark/LLaMA_HF/7B # decapoda-research/llama-7b-hf
            export DATA_NAME=kegg # string, kegg, indra, kbase
            export TASK=entity # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase)
            export TEST_BATCH_SIZE=8
            export TRAIN_BATCH_SIZE=32 # used in finetuning
            export N_SHOTS=3
            export LORA_WEIGHTS=tloen/alpaca-lora-7b

            # python ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --test_batch_size $TEST_BATCH_SIZE \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --n_shots $N_SHOTS
            
            : '
                --load_8bit
            '
            
            break
            ;;
        "Run LLaMA-2")
            echo "you chose Run LLaMA-2."
            
            export MODEL_NAME=LLaMA-2
            export MODEL_TYPE=meta-llama/Llama-2-7b-chat-hf
            # export MODEL_TYPE=meta-llama/Llama-2-13b-chat-hf
            # export MODEL_TYPE=meta-llama/Llama-2-70b-chat-hf
            # export MODEL_TYPE=LoftQ/Llama-2-13b-hf-4bit-64rank # LoftQ test
            export DATA_NAME=kbase # string, kegg, indra, kbase, lll
            export TASK=entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-2-7b-chat-hf/string/entity/final_merged
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-2-7b-chat-hf/string/entity/final_checkpoint/
            export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-2-7b-chat-hf/kegg/entity/final_checkpoint
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-2-70b-chat-hf/string/entity/final_checkpoint
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-2-7b-chat-hf/kbase/entity_type/final_checkpoint/
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS
                
            
            : '
            # export MODEL_TYPE=meta-llama/Llama-2-7b-hf
            # export MODEL_TYPE=upstage/Llama-2-70b-instruct-v2
            
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run LLaMA-3")
            echo "you chose Run LLaMA-3."
            
            export MODEL_NAME=LLaMA-3
            # export MODEL_TYPE=meta-llama/Meta-Llama-3-8B
            # export MODEL_TYPE=meta-llama/Meta-Llama-3-8B-Instruct
            export MODEL_TYPE=meta-llama/Meta-Llama-3-70B
            # export MODEL_TYPE=meta-llama/Meta-Llama-3-70B-Instruct
            export DATA_NAME=kbase # string, kegg, indra, kbase, lll
            export TASK=entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/meta-llama/Llama-3-8B/kegg/entity/final_checkpoint
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                
            
            : '
            # export MODEL_TYPE=meta-llama/Llama-2-7b-hf
            # export MODEL_TYPE=upstage/Llama-2-70b-instruct-v2
            
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run LLaMA-3.1")
            echo "you chose Run LLaMA-3.1"
            
            export MODEL_NAME=LLaMA-3.1
            export MODEL_TYPE=meta-llama/Meta-Llama-3.1-8B-Instruct
            # export MODEL_TYPE=meta-llama/Meta-Llama-3.1-70B-Instruct
            # export MODEL_TYPE=meta-llama/Meta-Llama-3.1-405B-Instruct
            # export MODEL_TYPE=meta-llama/Meta-Llama-3.1-8B
            # export MODEL_TYPE=meta-llama/Meta-Llama-3.1-70B
            # export MODEL_TYPE=meta-llama/Meta-Llama-3.1-405B
            export DATA_NAME=kbase
            export TASK=entity_and_entity_type # entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=1
            export N_SHOTS=0
            export MAX_NEW_TOKENS=2000 # 500, 2000
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --max_new_tokens $MAX_NEW_TOKENS \
                --n_shots $N_SHOTS \
                --use_quantization
                
            : '
                # if you run into OOM, try quantization.
                --use_quantization \
            '
            
            break
            ;;
        "Run Mistral")
            echo "you chose Run Mistral."
            
            export MODEL_NAME=Mistral
            export MODEL_TYPE=mistralai/Mistral-7B-Instruct-v0.2
            # export MODEL_TYPE=mistralai/Mixtral-8x7B-Instruct-v0.1
            export DATA_NAME=kbase # string, kegg, indra, kbase
            export TASK=entity_and_entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=2
            export N_SHOTS=0
            export MAX_NEW_TOKENS=2000 # 500, 2000
            export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/mistralai/Mistral-7B-Instruct-v0.2/kbase/entity_type
            # export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/mistralai/Mixtral-8x7B-Instruct-v0.1/kbase/entity_type
            
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --max_new_tokens $MAX_NEW_TOKENS \
                --n_shots $N_SHOTS \
                --use_quantization
                
            
            : '
            # export MODEL_TYPE=meta-llama/Llama-2-7b-hf
            # export MODEL_TYPE=upstage/Llama-2-70b-instruct-v2
            
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run Solar")
            echo "you chose Run Solar."
            
            export MODEL_NAME=Solar
            # export MODEL_TYPE=upstage/SOLAR-10.7B-v1.0
            export MODEL_TYPE=upstage/SOLAR-10.7B-Instruct-v1.0
            export DATA_NAME=kbase # string, kegg, indra, kbase
            export TASK=entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/tiiuae/falcon-7b/kbase/entity_type/final_checkpoint/

            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
            
            : '
            # export MODEL_TYPE=meta-llama/Llama-2-7b-hf
            # export MODEL_TYPE=upstage/Llama-2-70b-instruct-v2
            
                --lora_finetune \
                --lora_output_dir $LORA_OUTPUT_DIR
                --use_quantization \
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run Falcon")
            echo "you chose Run Falcon."
            
            export MODEL_NAME=Falcon
            # export MODEL_TYPE=tiiuae/falcon-7b
            export MODEL_TYPE=tiiuae/falcon-40b
            # export MODEL_TYPE=tiiuae/falcon-7b-instruct
            # export MODEL_TYPE=tiiuae/falcon-40b-instruct
            export DATA_NAME=kbase # string, kegg, indra, kbase
            export TASK=entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/tiiuae/falcon-7b/kbase/entity_type/final_checkpoint/
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
                
            break
            ;;
        "Run MPT")
            echo "you chose Run MPT."
            
            export MODEL_NAME=MPT
            # export MODEL_TYPE=mosaicml/mpt-7b-chat
            export MODEL_TYPE=mosaicml/mpt-30b-chat
            export DATA_NAME=kbase # string, kegg, indra, kbase
            export TASK=entity_type # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase), entity_and_entity_type (kbase)
            export TRAIN_BATCH_SIZE=2 # used in finetuning
            export VALIDATION_BATCH_SIZE=8 # used in finetuning
            export TEST_BATCH_SIZE=8
            export N_SHOTS=0
            export LORA_WEIGHTS=/scratch/ac.gpark/LoRA_finetuned_models/mosaicml/mpt-7b-chat/kbase/entity_type/final_checkpoint/
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # torchrun --nproc_per_node=4 ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --validation_batch_size $VALIDATION_BATCH_SIZE \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS \
                --lora_finetune \
                --use_quantization \
                --lora_output_dir $LORA_OUTPUT_DIR
            
            : '
            # export MODEL_TYPE=meta-llama/Llama-2-7b-hf
            # export MODEL_TYPE=upstage/Llama-2-70b-instruct-v2
            
                --lora_finetune \
                --lora_output_dir $LORA_OUTPUT_DIR
                --lora_weights $LORA_WEIGHTS
            '
            
            break
            ;;
        "Run RST")
            echo "you chose Run RST."
            
            export MODEL_NAME=RST
            export MODEL_TYPE=XLab/rst-all-11b
            export DATA_NAME=string # string, kegg, indra, kbase
            export TASK=entity # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase)
            export TEST_BATCH_SIZE=16
            export TRAIN_BATCH_SIZE=32 # used in finetuning
            export N_SHOTS=5

            python ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --test_batch_size $TEST_BATCH_SIZE \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --n_shots $N_SHOTS
                
            break
            ;;
        "Run BioGPT")
            echo "you chose Run BioGPT."
            
            export MODEL_NAME=BioGPT
            export MODEL_TYPE=microsoft/BioGPT-Large # microsoft/biogpt, microsoft/BioGPT-Large, microsoft/BioGPT-Large-PubMedQA
            export DATA_NAME=string # string, kegg, indra, kbase
            export TASK=entity # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase)
            export TEST_BATCH_SIZE=32
            export TRAIN_BATCH_SIZE=32 # used in finetuning
            export N_SHOTS=5

            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --test_batch_size $TEST_BATCH_SIZE \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --n_shots $N_SHOTS
                
            break
            ;;
        "Run BioMedLM")
            echo "you chose Run BioMedLM."
            
            export MODEL_NAME=BioMedLM
            export MODEL_TYPE=stanford-crfm/BioMedLM
            export DATA_NAME=kegg # string, kegg, indra, kbase
            export TASK=entity # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase)
            export TEST_BATCH_SIZE=32
            export TRAIN_BATCH_SIZE=32 # used in finetuning
            export N_SHOTS=1
            
            # python ~/BioIE-LLM-WIP/src/run_model.py \
            # accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
            
            accelerate launch ~/BioIE-LLM-WIP/src/run_model.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --test_batch_size $TEST_BATCH_SIZE \
                --train_batch_size $TRAIN_BATCH_SIZE \
                --n_shots $N_SHOTS
                
            break
            ;;
        "Run Gemini")
            echo "you chose Run Gemini."
            
            export MODEL_NAME=Gemini
            export MODEL_TYPE=gemini-pro
            export DATA_NAME=kegg # string, kegg, indra, kbase
            export TASK=entity # entity (string, kegg), relation (string, kegg), relation_type (indra), entity_type (kbase)
            export TEST_BATCH_SIZE=1
            export N_SHOTS=2

            python ~/BioIE-LLM-WIP/src/run_gemini.py \
                --model_name $MODEL_NAME \
                --model_type $MODEL_TYPE \
                --data_repo_path $DATA_REPO_PATH \
                --output_dir $OUTPUT_DIR \
                --data_name $DATA_NAME \
                --task $TASK \
                --test_batch_size $TEST_BATCH_SIZE \
                --n_shots $N_SHOTS
            
            break
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
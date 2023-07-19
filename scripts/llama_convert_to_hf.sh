#!/bin/bash

export CONVERT_SCRIPT=/lambda_stor/homes/ac.gpark/anaconda3/envs/llama/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py
export INPUT_DIR=/scratch/ac.gpark/LLaMA
export OUTPUT_DIR=/scratch/ac.gpark/LLaMA_HF/7B

python $CONVERT_SCRIPT \
    --input_dir $INPUT_DIR \
	--model_size 7B \
	--output_dir $OUTPUT_DIR
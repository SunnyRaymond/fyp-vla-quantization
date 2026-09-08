set -ex


export OPENAI_API_KEY=YOUR_OPENAI_KEY

FINAL_RUN_NAME=$1

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava_hf --model_args pretrained=${FINAL_RUN_NAME},attn_implementation=eager,device_map=auto,dtype=bfloat16 --tasks mmmu_val --batch_size 1 --log_samples --log_samples_suffix mmmu_val --output_path ${FINAL_RUN_NAME}/mmmu_val &
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 -m lmms_eval --model llava_hf --model_args pretrained=${FINAL_RUN_NAME},attn_implementation=eager,device_map=auto,dtype=bfloat16 --tasks ai2d --batch_size 1 --log_samples --log_samples_suffix ai2d --output_path ${FINAL_RUN_NAME}/ai2d &
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 -m lmms_eval --model llava_hf --model_args pretrained=${FINAL_RUN_NAME},attn_implementation=eager,device_map=auto,dtype=bfloat16 --tasks seedbench_2_plus --batch_size 1 --log_samples --log_samples_suffix seedbench_2_plus --output_path ${FINAL_RUN_NAME}/seedbench_2_plus &
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava_hf --model_args pretrained=${FINAL_RUN_NAME},attn_implementation=eager,device_map=auto,dtype=bfloat16 --tasks mmstar --batch_size 1 --log_samples --log_samples_suffix mmstar --output_path ${FINAL_RUN_NAME}/mmstar &
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava_hf --model_args pretrained=${FINAL_RUN_NAME},attn_implementation=eager,device_map=auto,dtype=bfloat16 --tasks seedbench --batch_size 1 --log_samples --log_samples_suffix seedbench --output_path ${FINAL_RUN_NAME}/seedbench &
wait
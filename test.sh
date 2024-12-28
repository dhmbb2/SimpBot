
echo "using checkpoint from $1"

CUDA_VISIBLE_DEVICES=0 opencompass \
    --datasets mmlu_ppl hellaswag_clean_ppl winogrande_ll ARC_e_ppl ARC_c_clean_ppl SuperGLUE_BoolQ_few_shot_ppl \
    --summarizer example \
    --hf-type base \
    --hf-path $1 \
    --tokenizer-kwargs padding_side="left" truncation="left" \
    --max-seq-len 1024 \
    --batch-size 4 \
    --hf-num-gpus 1 \
    --work-dir "./test_result" \
    --debug

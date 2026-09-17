#################################
# HuggingFace: `https://huggingface.co/intfloat/e5-mistral-7b-instruct`
# Sample vLLM Deployment on L40S:
#################################
```
export HF_TOKEN=<your-huggingface-token>
pip install vllm==0.19.0 mistral-common==1.9.1 tqdm==4.67.3 jupyter==1.1.1 hf_transfer==0.1.9 transformers==4.55.2
nohup python -m vllm.entrypoints.openai.api_server \
    --model=intfloat/e5-mistral-7b-instruct \
    --runner pooling \
    --dtype float16 \
    > vllm.log 2>&1 &
```


#################################
# Other
#################################
To deploy the embedding model as a standalone job, run `make deploy-embedding-model`.

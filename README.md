# fairseq2-llama
<p align="center">
  <img src="https://github.com/Efrainq07/fairseq2-llama/assets/33973526/4067c926-c5ed-46e9-a037-bc3a74e9f423" alt="LLaMA image"/>
</p>

**Training scripts and data loaders for Fairseq2 LLaMA implementation.**

# Download Model Weights
Run the `download_model.sh` script to download the LLaMA 2 tokenizer along with the [TinyLLaMA](https://github.com/jzhang38/TinyLlama) weights. 
```
bash download_model.sh
```
After that run the `weight_converter.py` script to convert the TinyLLaMA weights to the Fairseq2 format.
```
python weight_converter.py
```

# Test the Model Inference
Run the test script and see that everything works fine:
```
python test.py
```

# Finetune Model
Run the trainer script and start finetuning the TinyLLaMA model, notice that finetuning dataset is [WikiText103](https://paperswithcode.com/dataset/wikitext-103)
```
python trainer.py
```

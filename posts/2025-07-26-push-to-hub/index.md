
---
title: TIL&#58; Custom Composer Callback to Push Checkpoints to HuggingFace Hub During Training. 
date: "2025-07-26"
author: Vishal Bakshi
description: In this short TIL blog post, I'm going to share the code I wrote with Claude's help for a custom Composer callback which pushes the model to Hugging Face Hub every specified number of steps. The purpose of doing so is so that you can run evaluation after training so it doesn't slow down training. 
filters:
   - lightbox
lightbox: auto
categories:
    - LLM
    - Custom Composer Callback
---

## Background 

In this short TIL blog post, I'm going to share the code I wrote with Claude's help for a custom Composer callback which pushes the model to Hugging Face Hub every specified number of steps. The purpose of doing so is so that you can run evaluation after training so it doesn't slow down training. 

## Custom Composer Callback 

```python
class HFPushCallback(Callback):
  def __init__(self, repo_id: str = "LocalResearchGroup/push-to-hub-test", push_every_n_steps: int = 10):
      self.repo_id = repo_id
      self.push_every_n_steps = push_every_n_steps
      self.token = os.getenv("HF_TOKEN")
      self.hf_api = HfApi(token=self.token)
  
      create_repo(
          repo_id=self.repo_id,
          token=self.token,
          private=True,
          exist_ok=True
      )
  
  def batch_end(self, state: State, logger: Logger) -> None:            
      if state.timestamp.batch.value % self.push_every_n_steps == 0:
          self._push_model(state)
  
  def _push_model(self, state: State):
      with tempfile.TemporaryDirectory() as temp_dir:
          state.model.model.save_pretrained(temp_dir)
          
          self.hf_api.upload_folder(
              folder_path=temp_dir,
              repo_id=self.repo_id,
              commit_message=f"Step {state.timestamp.batch.value}"
          )
```

Important to note, `state.model` is the `ComposerHFCausalLM` wrapper around the HuggingFace model, so you have to access `state.model.model` to use the attribute `save_pretrained`. 

## Running Inference

You can use the following code to run inference on the model, just as you would any set of PEFT adapters from Hugging Face Hub. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


model_id = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

model = PeftModel.from_pretrained(
    model,
    "<repo_id>",
    revision = "<revision>"
)

prompt = "The best thing about artificial intelligence is "
inputs = tokenizer(prompt, return_tensors="pt")
attention_mask = inputs["attention_mask"]

outputs = model.generate(
  inputs['input_ids'],
  attention_mask=attention_mask,
  pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))      
```

The `revision` parameter is the commit ID in your Hugging Face repo. In this way, if you, say, push your model every 100 steps, then you can use the `revision` argument for each of those checkpoints and run your evaluations. Then you can log those evaluations to your W&B project so that your evaluation log is comparable with other training logs. 
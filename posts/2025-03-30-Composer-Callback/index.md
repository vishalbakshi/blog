
---
title: TIL&#58; Creating a Custom Composer Callback 
date: "2025-03-30"
author: Vishal Bakshi
description: A walkthrough of my first custom Composer callback where I log weight, activation, gradient and loss data types during the training loop.
filters:
   - lightbox
lightbox: auto
categories:
    - LLM
    - deep learning
---

## Background

I'm learning to use LLM-Foundry to finetune SLMs. To better understand what's going on in the training loop when using Flash Attention 2 (for SmolLM2-135M), I decided to ask Claude to write me a custom callback. Here is my [full Claude conversation](https://claude.ai/share/9bb6c135-2ffb-42be-91bc-b4e4a6356173).

## Initial Plan

At first, I was planning to fork Composer (which I did), create a new branch for edits (print statements of datatypes in the `Trainer` code), and install that repo/branch for training. However, as I was chatting with Claude, it offered an option to write a callback instead. Being that [this is a core philosophy of how Composer is built](https://docs.mosaicml.com/projects/composer/en/stable/getting_started/welcome_tour.html#:~:text=This%20is%20based%20on%20the%20two%2Dway%20callback%20system%20from%20(Howard%20et%20al%2C%202020)), it was a no brainer for me to pursue.

## First Callback

The first callback Claude wrote (I guided it a little bit by feeding it Composer's `trainer.py` and giving it their callback example from the docs) was as follows:

```python
class WeightDtypeMonitor(Callback):
    def __init__(self, backward_log_interval=5):
        self.backward_log_interval = backward_log_interval
    
    def fit_start(self, state: State, logger: Logger) -> None:
        self._log_dtypes(state, logger, "fit_start")
    
    def after_backward(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.backward_log_interval == 0:
            self._log_dtypes(state, logger, f"backward_{state.timestamp.batch.value}")
    
    def epoch_end(self, state: State, logger: Logger) -> None:
        self._log_dtypes(state, logger, f"epoch_{state.timestamp.epoch.value}")
    
    def _log_dtypes(self, state: State, logger: Logger, prefix: str) -> None:
        model = state.model
        logger.log_metrics({
            f"dtype/{prefix}/lm_head": str(model.model.base_model.model.lm_head.weight.dtype),
            f"dtype/{prefix}/q_proj_base": str(model.model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.dtype),
            f"dtype/{prefix}/q_proj_lora_A": str(model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.dtype),
            f"dtype/{prefix}/q_proj_lora_B": str(model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.dtype)
        })
```

For reference, here is the list of events available in Composer during the training loop:

```
# <INIT>
# <BEFORE_LOAD>
# <AFTER_LOAD>
# <FIT_START>
for epoch in range(NUM_EPOCHS):
    # <EPOCH_START>
    while True:
        # <BEFORE_DATALOADER>
        batch = next(dataloader)
        if batch is None:
            break
        inputs, targets = batch
        # <AFTER_DATALOADER>

        # <BATCH_START>

        # <BEFORE_FORWARD>
        outputs = model.forward(inputs)
        # <AFTER_FORWARD>

        # <BEFORE_LOSS>
        loss = model.loss(outputs, targets)
        # <AFTER_LOSS>

        # <BEFORE_BACKWARD>
        loss.backward()
        # <AFTER_BACKWARD>

        optimizer.step()
        optimizer.zero_grad()

        # <BATCH_END>
    # <EPOCH_END>
```

For each event I wanted to log data types for, the callback passes `state` (where the `model` is stored), `logger` (to do the logging) and a `prefix` (to denote what's being logged). Only every `backward_log_interval`-th batch's backward pass is logged, to avoid clutter.

Here is example output:

```
# fit_start
Train dtype/fit_start/lm_head: "torch.float32"
Train dtype/fit_start/q_proj_base: "torch.float32"
Train dtype/fit_start/q_proj_lora_A: "torch.float32" 
Train dtype/fit_start/q_proj_lora_B: "torch.float32"

# after_backward
Train dtype/backward_0/lm_head: "torch.float32"
Train dtype/backward_0/q_proj_base: "torch.float32"
Train dtype/backward_0/q_proj_lora_A: "torch.float32"
Train dtype/backward_0/q_proj_lora_B: "torch.float32"
```

I was surprised to see that everything was in float32, I thought Flash Attention 2 (FA2) used mixed precision? Note that I haven't read the FA2 paper.

## Second Callback

Now that I understood a basic logging callback, I asked Claude to generate a callback that would help me see where mixed precision came into play. This one was a bit more involved:

```python
class DtypeMonitor(Callback):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.hooks = []
    
    def fit_start(self, state: State, logger: Logger) -> None:
        self._log_weight_dtypes(state, logger, "fit_start")
    
    def before_forward(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval == 0:
            # Log input tensor dtypes
            if isinstance(state.batch, dict) and 'input_ids' in state.batch:
                logger.log_metrics({
                    "dtype/input/input_ids": str(state.batch['input_ids'].dtype)
                })
            
            # Register hooks to capture activation dtypes
            layer = state.model.model.base_model.model.model.layers[0].self_attn
            
            def hook_fn(name):
                def _hook(module, inputs, outputs):
                    # Log input activation dtype
                    if isinstance(inputs, tuple) and len(inputs) > 0:
                        logger.log_metrics({f"dtype/activation/{name}_input": str(inputs[0].dtype)})
                    
                    # Log output activation dtype
                    if isinstance(outputs, torch.Tensor):
                        logger.log_metrics({f"dtype/activation/{name}_output": str(outputs.dtype)})
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        logger.log_metrics({f"dtype/activation/{name}_output": str(outputs[0].dtype)})
                return _hook
            
            # Clear old hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            
            # Register new hooks
            self.hooks.append(layer.q_proj.register_forward_hook(hook_fn("q_proj")))
            self.hooks.append(layer.register_forward_hook(hook_fn("self_attn")))
    
    def after_forward(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval == 0:
            # Log model output dtype
            if isinstance(state.outputs, torch.Tensor):
                logger.log_metrics({
                    "dtype/computation/output": str(state.outputs.dtype)
                })
    
    def after_loss(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval == 0:
            # Log loss dtype
            if isinstance(state.loss, torch.Tensor):
                logger.log_metrics({
                    "dtype/computation/loss": str(state.loss.dtype)
                })
            elif isinstance(state.loss, dict) and 'total' in state.loss:
                logger.log_metrics({
                    "dtype/computation/loss": str(state.loss['total'].dtype)
                })
    
    def after_backward(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval == 0:
            self._log_weight_dtypes(state, logger, f"backward_{state.timestamp.batch.value}")
            
            # Check gradient dtypes
            model = state.model
            lora_A = model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default
            if hasattr(lora_A, 'weight') and lora_A.weight.grad is not None:
                logger.log_metrics({
                    "dtype/gradient/q_proj_lora_A": str(lora_A.weight.grad.dtype)
                })
    
    def epoch_end(self, state: State, logger: Logger) -> None:
        self._log_weight_dtypes(state, logger, f"epoch_{state.timestamp.epoch.value}")
        # Remove any remaining hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _log_weight_dtypes(self, state: State, logger: Logger, prefix: str) -> None:
        model = state.model
        logger.log_metrics({
            f"dtype/{prefix}/lm_head": str(model.model.base_model.model.lm_head.weight.dtype),
            f"dtype/{prefix}/q_proj_base": str(model.model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.dtype),
            f"dtype/{prefix}/q_proj_lora_A": str(model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.dtype),
            f"dtype/{prefix}/q_proj_lora_B": str(model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight.dtype)
        })
```

Fortunately, I had just recently learned about `register_forward_hook` and created a short TIL video about it:


{{< video https://www.youtube.com/embed/Y6qgWxU3oO4 >}}

In short, `register_forward_hook` exposes the forward pass inputs and outputs. You can manipulate both but you have access to inputs/outputs _after_ the forward pass so you can't change the inputs before they go into the forward pass. Thankfully that restriction doesn't matter in my case, as I only want to log data types.

Running the training loop with this callback generated the following logs:

```
 Train dtype/input/input_ids: "torch.int64"
 Train dtype/activation/q_proj_input: "torch.float32"
 Train dtype/activation/q_proj_output: "torch.bfloat16"
 Train dtype/activation/self_attn_output: "torch.bfloat16"
 Train dtype/computation/loss: "torch.float32"
 Train dtype/backward_0/lm_head: "torch.float32"
 Train dtype/backward_0/q_proj_base: "torch.float32"
 Train dtype/backward_0/q_proj_lora_A: "torch.float32"
 Train dtype/backward_0/q_proj_lora_B: "torch.float32"
 Train dtype/gradient/q_proj_lora_A: "torch.float32"
```

This shed some more light into what's going on! The inputs to `q_proj` is float32 but the outputs are bfloat16. The loss and gradients are both in float32.

## Final Thoughts

This exercise has blown up the possibilities available to me for better understanding what goes on during training! I have only gotten a cursory glimpse at the internal mechanism of mixed precision training, but it's relatively simple for me take this a step further by analyzing more data types during all training events for all model components. That'll be a future blog post or video this week. 

Thanks for reading! Lots more content on my [YouTube channel](https://www.youtube.com/@vishal_learner) that I'm working on growing this year so please subscribe to stay in the loop.
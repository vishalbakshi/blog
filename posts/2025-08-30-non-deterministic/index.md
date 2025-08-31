---
title: The Term "Non-Deterministic" and LLMs
date: "2025-08-30"
author: Vishal Bakshi
description: I don't think "non-deterministic" is a most precise way of describing LLMs.
filters:
   - lightbox
lightbox: auto
categories:
    - LLM
---

I have recently found myself using the terms "non-deterministic" to describe LLM behavior. However, something feels off about using that term and I'm nearly convinced that not only is it (sometimes) incorrect, it is imprecise, as it leaves unexplained a critical charactericistic of LLM behavior that makes LLMs different from deterministic functions.

First, defining "deterministic algorithm" (Wikipedia):

> In computer science, a deterministic algorithm is an algorithm that, given a particular input, will always produce the same output, with the underlying machine always passing through the same sequence of states.

LLMs can be deterministic (i.e. temperature = 0, `do_sample=False`). For example running the following code passes all 100 assertions:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The best thing about artificial intelligence is "
inputs = tokenizer(prompt, return_tensors="pt")
attention_mask = inputs["attention_mask"]

texts = []
for _ in range(100):
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    texts.append(text)

for text in texts: assert text == "The best thing about artificial intelligence is  that it can be used to solve problems that would otherwise be impossible to solve.\n\nFor"
```

So, LLMs can be deterministic. If temperature is not zero and other sampling approaches are used then yes, LLMs are non-deterministic.

What I think people mean by saying "LLMs are non-deterministic" is something like the following from the [Steering Semantic Data Processing With DocWrangler](https://arxiv.org/abs/2504.14764) paper by Shreya Shankar, et al:

> users need to understand their data to write effective pipelines, yet they need to construct pipelines to extract the data necessary for that understanding

Thinking on that a bit more, what I think people mean by saying "LLMs are non-deterministic" is: what inputs to give LLMs for a desired output is ambiguous. Prompt engineering being a thing is a great example of this. I don't know enough mathematics to know if there's a term for this. "Input ambiguous"? "Non-deterministic on both ends"? The best Sonnet 4 came up with was "non-invertible" (other options was non-transparent). GPT-5 Thinking came up with a more sophisticated response _Prompting LLMs is an ill-posed inverse problem._

> - Inverse problem: you start from a desired output and try to find an input (prompt) that yields it.
> - Ill-posed (Hadamard): the inverse fails one or more of
>   - existence (your target may be unreachable),
>   - uniqueness (many prompts produce similar outputs → non-injective),
>   - stability (tiny prompt tweaks swing the output a lot).
> 
> Separately, decoding can be stochastic (temperature/top-p), which is where “non-deterministic” actually applies. With temperature=0 and deterministic kernels, the model is deterministic—but the inverse remains ill-posed.

A well-posed problem (Wiki):

> In mathematics, a well-posed problem is one for which the following properties hold:
> 
> 1. The problem has a solution
> 2. The solution is unique
> 3. The solution's behavior changes continuously with the initial conditions.

Problems we try to solve with LLMs often fail all three properties, but again, I don't know enough about mathematics to know if this truly applies to LLMs.

Most of my interactions with LLMs are through Claude Projects for coding assistance, and I make sure I understand the code (and that it works) before using it, so input ambiguity is acceptable.  As I learn to use LLMs to build pipelines, the input ambiguity problem sharpens, and quickly makes my pipeline brittle. Over the next couple weeks, I plan on learning more about DocWrangler and DSPy to better understand how to temper my pipeline. 




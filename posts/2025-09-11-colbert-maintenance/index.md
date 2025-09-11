---
title: Batch Size Causes `BertModel` Forward Pass Divergence Between `torch==2.0.1` and `torch==2.1.0` for `colbert-ai`.
date: "2025-09-11"
author: Vishal Bakshi
description: In this blog post I document as far as I could get in determining what caused a `BertModel` forward pass divergence between PyTorch versions `2.0.1` and `2.1.0`. Certain batch sizes yield different model layer outputs between PyTorch version, while other batch sizes don't.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

I've recently been documenting how PyTorch version changes impact stanford-futuredata/ColBERT (`colbert-ai` on PyPI) intermediate and final index artifacts. The index artifact I'll focus on in this blog post is the very important [`local_sample_embs`](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/collection_indexer.py#L137) tensor. This is the sample of token embeddings used to calculate the centroids, which are later on used during search. Instead of loading and comparing full document token embeddings, ColBERT's PLAID index compares centroid IDs (integers) and compressed residuals (low bit vectors) in the first three stages of the search pipeline, only decompressing residuals in the final stage. This reduces storage footprint and search latency. 

When comparing `local_sample_embs` (`torch.float16`) between `torch==2.4.1` and `torch==2.5.0`, using atol=1e-4 and rtol=1e-3 in `torch.allclose`:

```
torch.allclose: False
Mean Acc:       0.7978946566581726      
MAD:            1.2740434613078833e-05  
Max Abs Diff:   0.00115966796875 
```

## Whats the Diff?

What's causing the `local_sample_embs` to be different across PyTorch versions? Here's how I explored it:

`colbert-ai` encodes passages in batches (1600 at a time in my case, for a total of 29 batches across 46107 passages) so I compared model layer outputs for each batch between PyTorch versions using the following script:

```python
checkpoint = Checkpoint("answerdotai/answerai-colbert-small-v1", colbert_config=config)
sample_pids = torch.load(f"{MOUNT}/{project}/{date}-{source}-{nranks}/sample_pids.pt")

idx = 0
for idx in range(29):
    docs = passages['text'][list(sample_pids)[1600*idx:1600*(idx+1)]]
    text_batches, reverse_indices = checkpoint.doc_tokenizer.tensorize(docs, bsize=config.index_bsize)
    input_ids = text_batches[0][0] 
    attention_mask = text_batches[0][1] 

    with torch.cuda.amp.autocast():
        outputs_dict = {}
        def capture_output(name):
            def hook_fn(module, input, output):
                outputs_dict[name] = output[0].detach()
            return hook_fn

        hooks = []
        for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
        with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
        for h in hooks: h.remove()

        torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/lse_outputs_dict_{idx}.pt")
        print(f"lse_outputs_dict_{idx} saved!")
```

Batch idx `2`, `3`, `4`, `5`, `12`, `15`, `18`, `19`, `20`, and `26` fail the `torch.allclose` comparison between `BertModel` layer outputs. Why is that the case?

Here are the tensor shapes for each batch of `input_ids`:

```
0 torch.Size([32, 71])
1 torch.Size([32, 72])
2 torch.Size([32, 79])   # fails
3 torch.Size([32, 78])   # fails
4 torch.Size([32, 77])   # fails
5 torch.Size([32, 194])  # fails
6 torch.Size([32, 70])
7 torch.Size([32, 73])
8 torch.Size([32, 71])
9 torch.Size([32, 68])
10 torch.Size([32, 66])
11 torch.Size([32, 115])
12 torch.Size([32, 82])  # fails
13 torch.Size([32, 115])
14 torch.Size([32, 115])
15 torch.Size([32, 80])  # fails
16 torch.Size([32, 72])
17 torch.Size([32, 64])
18 torch.Size([32, 90])  # fails
19 torch.Size([32, 82])  # fails
20 torch.Size([32, 86])  # fails
21 torch.Size([32, 63])
22 torch.Size([32, 71])
23 torch.Size([32, 62])
24 torch.Size([32, 61])
25 torch.Size([32, 67])
26 torch.Size([32, 83])  # fails
27 torch.Size([32, 69])
28 torch.Size([32, 72])
```

The batches that diverge have a second dimension of: `79`, `78`, `77`, `194`, `82`, `80`, `90`, `86`, and `83`.

The batches that do not diverge have a second dimension of: `71`, `72`, `70`, `73`, `68`, `66`, `115`, `64`, `63`, `62`, `61`, `67`, `69`.

It is interesting to note that these sets do not intersect. To test if batch size is the root cause, I index into the first 70 items of the diverging batches and run the layer output comparison again:

```python
batch_idx = 70
for idx in [2, 3, 4, 5, 12, 15, 18, 19, 20, 26]:
    docs = passages['text'][list(sample_pids)[1600*idx:1600*(idx+1)]]
    text_batches, reverse_indices = checkpoint.doc_tokenizer.tensorize(docs, bsize=config.index_bsize)
    input_ids = text_batches[0][0][:, :batch_idx]
    attention_mask = text_batches[0][1][:, :batch_idx]

    with torch.cuda.amp.autocast():
        outputs_dict = {}
        def capture_output(name):
            def hook_fn(module, input, output):
                outputs_dict[name] = output[0].detach()
            return hook_fn

        hooks = []
        for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
        with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
        for h in hooks: h.remove()

        torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/lse_outputs_dict_{idx}.pt")
        print(f"lse_outputs_dict_{idx} saved!")
```

```python
for idx in range(29):
    a = torch.load(f"{MOUNT}/{root_a}/lse_outputs_dict_{idx}.pt")
    b = torch.load(f"{MOUNT}/{root_b}/lse_outputs_dict_{idx}.pt")

    for i in range(len(a.keys())):
        a_ = a[f"{i}"]
        b_ = b[f"{i}"]
        assert _close(a_, b_)
```

All model layer outputs match between PyTorch versions! Just to be sure, I tried `batch_idx` of `61`, `64` and `68`, and all model layer outputs match.

## Closing Thoughts

Earlier today I read the [Thinking Machines blog post on why LLM inference is non-deterministic](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). The main cause for non-determinism is that not all tensor ops are _batch size invariant_:

> As it turns out, our request’s output does depend on the parallel user requests. Not because we’re somehow leaking information across batches — instead, it’s because our forward pass lacks “batch invariance”, causing our request’s output to depend on the batch size of our forward pass.

> To explain batch invariance, let’s simplify the system and look solely at matmuls. You can assume that all matmul implementations are “run-to-run deterministic."This is not totally true, but most common matmul implementations do have this property. However, they are not “batch-invariant.” In other words, when the batch size changes, each element in the batch can get different results.

While I'm not going to (can't?) dig into PyTorch to understand what is causing batch size variance between `2.4.1` and `2.5.0`, I think there is enough evidence to show that something in PyTorch is causing it. If you disagree with that conclusion, [please @ me on Twitter](https://x.com/vishal_learner)!
---
title: Comparing `colbert-ai` Artifacts Between PyTorch Versions 2.0.1 and 2.1.0
date: "2025-09-13"
author: Vishal Bakshi
description: The `BertModel` forward pass diverges between these two PyTorch versions, resulting in different document token embeddings and eventually, different final index artifacts. Swapping `local_sample_embs` from 2.0.1 to 2.1.0 yields identical index artifacts (except the sort order of centroid IDs).
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

I've been redoing my `colbert-ai` index comparisons between PyTorch versions using [bitsandbytes' `torch.allclose` tolerances](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/39dd8471c1c0677001d0d20ba2218b14bf18fd00/tests/test_optim.py#L189-L194). In this blog post I explore `colbert-ai` index artifact differences between PyTorch versions 2.0.1 and 2.1.0. 

## Comparing Intermediate and Final Index Artifacts

### Final Index Artifacts 

Using the more lenient bitsandbytes tolerances, <mark>`avg_residual.pt` and `bucket_weights.pt` pass `torch.allclose`</mark> while `bucket_cutoffs` and `centroids` do not.

#### Integer Tensors

|Artifact|Description|dtype|`torch.equal`|
|:-:|:-:|:-:|:-:|
|`codes.pt`|centroid id mapped to doc token embeddings|`torch.int32`|`False`|
|`residuals.pt`|difference b/w centroid and doc token embeddings|`torch.uint8`|`False`|
|`ivf.pid.pt` (ivf)|unique pids per centroid id|`torch.int32`|<mark>shape mismatch<mark>|
|`ivf.pid.pt` (ivf_lengths)|number of pids per centroid id|`torch.int64`|`False`

#### Float Tensors

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`avg_residual.pt`|Average difference b/w centroids and doc token embeddings|`torch.float16`|`False`|`True`|
|`buckets.pt` (`bucket_cutoffs`)|The quantization bins|`torch.float32`|`False`|`False`|
|`buckets.pt` (`bucket_weights`)|The quantization values for each bin|`torch.float16`|`False`|`True`|
|`centroids.pt`|Centroids of clustered sample doc token embeddings|`torch.float16`|`False`|`False`|

### Intermediate Index Artifacts

"Intermediate" artifacts are tensors saved in the middle of the indexing pipeline by adding `torch.save` calls in `/colbert/indexing/collection_indexer.py` and `/colbert/modeling/checkpoint.py`.

#### Integer Tensors

Some of the intermediate artifacts are not tensors so the equality column I'm titling "Equal" instead of `torch.equal`.

|Artifact|Description|dtype|Equal|
|:-:|:-:|:-:|:-:|
|`sample_pids.pt`|A sample of passage ids used to calculate centroids|`int`|`True`|
|`num_passages.pt`|Number of sampled passages|`int`|`True`|
|`doclens.pt`|List of number of tokens per document|`int`|`True`|

#### Float Tensors

Using the more lenient bitsandbytes tolerances, none of the `torch.allclose` calls pass.

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`local_sample_embs.pt`|Embeddings of sample document passages used to calculate centroids|`torch.float16`|`False`|`False`
|`sample.pt`|95% of the values from `local_sample_embs.half()`|`torch.float16`|`False`|`False`
|`sample_heldout.pt`|5% of the values from `local_sample_embs.half()`|`torch.float16`|`False`|`False`
|`batches.pt`|1 batch of encoded passages|`torch.float16`|`False`|`False`
|`D.pt`|sorted and reshaped `batches`|`torch.float16`|`False`|`False`

## Root Cause of Divergence: `BertModel` Forward Pass

`local_sample_embs` are a critical tensor in the ColBERT indexing process: this is the sample of document token embeddings used to calculate centroids. These centroids are later mapped (`ivf.pid.pt`) to document token IDs, allowing a smaller footprint (instead of storing full document token embeddings, we only have to store integer centroid IDs and low-bit residual vectors--the difference between centroids and document token embeddings), and more efficient search (we only consider those documents that are close to centroids that are close to the query tokens). `local_sample_embs` fails `torch.allclose` between PyTorch versions 2.0.1 and 2.1.0. This divergence then results in different `centroids.pt` and eventually different final indexes (`ivf.pid.pt`) between torch versions. To prove this, I injected 2.0.1's `local_sample_embs` into 2.1.0 and the resulting intermediate and final artifacts were identical.

`local_sample_embs` are created by passing the sample passages through the `CollectionEncoder.encode_passages` method which eventually passes them through the `Checkpoint.bert` model. Given the same inputs (the sample passage) the BERT model produces different outputs between PyTorch versions. I found that regardless of what the input tokens are, the `BertModel` outputs fail `torch.allclose`. 

Here's the code I used to capture model layer outputs:

```python
docs = ["a"]
kpoint.doc_tokenizer.tensorize(docs, bsize=config.index_bsize)
input_ids = text_batches[0][0] 
attention_mask = text_batches[0][1] 

outputs_dict = {}
def capture_output(name):
    def hook_fn(module, input, output):
        outputs_dict[name] = output[0].detach()
    return hook_fn

with torch.cuda.amp.autocast():
    hooks = []
    for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
    with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
    for h in hooks: h.remove()
    torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/amp_outputs_dict.pt")
    print("amp_outputs_dict saved!")
```

For `docs` I tried a single letter (`"a"`), a test sentence (`["test input"]`) and different batches from the UKPLab/DAPR/ConditionalQA document collection. In all cases, the model layer outputs between PyTorch versions failed `torch.allclose`.

As an aside, I also discovered that even after swapping `local_sample_embs` and obtaining final `ivf.pid.pt` tensors that passed `torch.allclose`, the intermediate `codes` (centroid IDs) were sorted differently between PyTorch versions. I have detailed that observation [in another blog post](https://vishalbakshi.github.io/blog/posts/2025-09-09-colbert-maintenance/) in which I also go on to show that even differently sorted `codes`, as long as they contain the right IDs, can result in the correct final `ivf` (unique passage IDs per centroid) and `ivf_lengths` (number of passage IDs per centroid).
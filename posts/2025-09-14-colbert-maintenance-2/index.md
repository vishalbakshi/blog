---
title: Re-evaluating `colbert-ai` Index Artifacts Between PyTorch Versions with Precision-Based `torch.allclose` Tolerances
date: "2025-09-14"
author: Vishal Bakshi
description: Analysis of ColBERT indexing differences (using bitsandbytes tolerances) between versions where `torch.allclose` returns `False`. This analysis also led to multiple deep dives that are linked as separate blog posts.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

I recently learned that it's best practice to use different `torch.allclose` tolerances based on the precision of the floating point value. As a reminder, `torch.allclose` uses absolute and relative tolerances as follows: 

`∣input_i − other_i∣ ≤ atol + rtol × ∣other_i∣`

[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/39dd8471c1c0677001d0d20ba2218b14bf18fd00/tests/test_optim.py#L189) uses the following heuristic:

```python
if dtype == torch.float32:
    atol, rtol = 1e-6, 1e-5
elif dtype == torch.bfloat16:
    atol, rtol = 1e-3, 1e-2
else: # float16
    atol, rtol = 1e-4, 1e-3
```

Full-precision (`float32`) has the lowest tolerance, followed by half-precision (`float16`) and then `bfloat16`. I've been using default tolerances in all my `torch.allclose` calls, regardless of precision (`atol` = `1e-08`, `rtol` = `1e-05`). Comparing these with bitsandbytes' tolerances, these default tolerances are:

- float32: 100x smaller for `atol` and the same for `rtol`
- float16: 10_000x smaller for `atol` and 100x smaller for `rtol`
- bfloat16: 100_000x smaller for `atol` and 1000x smaller for `rtol`

As you can see, the bitsandbytes tolerances are much more forgiving for lower precision, which intuitively makes sense. 

::: {.callout-tip}
## Two Goals of this Blog Post
The first question I'll explore in this post: how does changing my `torch.allclose` tolerances affect index artifact comparison? In other words, are there tensors between PyTorch versions whose difference is larger than `atol + rtol × ∣other_i∣` when using bitsandbytes' more forgiving tolerances?

The second question I'll answer: when `torch.allclose` fails, what is the root cause?
:::

::: {.callout-important}
I use the full (69_199 documents) [UKPLab/DAPR/ConditionalQA](https://huggingface.co/datasets/UKPLab/dapr) dataset in this exercise. In previous blog posts I used a 1000-document subset.
:::

## Comparing All Consecutive Versions

In this section I'll document tensor shape mismatches and `torch.allclose` values (with default tolerances in the "Default" column and bitsandbytes tolerances in the "bnb" column) for tensor index artifacts between consecutive PyTorch versions from 1.13.1 (the version pinned in the latest `colbert-ai` release) to 2.8.0 (the latest PyTorch version available as of 9/14/2025). 

|PyTorch Version A|PyTorch Version B|All Shapes Match|Default|bnb|
|:-:|:-:|:-:|:-:|:-:|
|1.13.1|2.0.0|Yes|`True`|`True`|
|2.0.0|2.0.1|Yes|`True`|`True`|
|2.0.1|2.1.0|No (11/12 Match)|`False` (0/12 Match)|`False` (2/12 Match)
|2.1.0|2.1.1|Yes|`True`|`True`|
|2.1.1|2.1.2|Yes|`True`|`True`|
|2.1.2|2.2.0|Yes|`True`|`True`|
|2.2.0|2.2.1|Yes|`True`|`True`|
|2.2.1|2.2.2|Yes|`True`|`True`|
|2.2.2|2.3.0|Yes|`True`|`True`|
|2.3.0|2.3.1|Yes|`True`|`True`|
|2.3.1|2.4.0|Yes|`True`|`True`|
|2.4.0|2.4.1|Yes|`True`|`True`|
|2.4.1|2.5.0|No (11/12 Match)|`False` (0/12 Match)|`False` (2/12 Match)|
|2.5.0|2.5.1|Yes|`True`|`True`|
|2.5.1|2.6.0|Yes|`True`|`True`|
|2.6.0|2.7.0|Yes|`True`|`True`|
|2.7.0|2.7.1|Yes|`True`|`True`|
|2.7.1|2.8.0|Yes|`False` (8/12 Match)|`False` (9/12 Match)|

In the three version comparisons where `torch.allclose` failed using default `atol` and `rtol` values, using bitsandbytes values yielded the same overall result (not all tensors match) but with two more matches for 2.0.1 --> 2.1.0 and 2.4.1 --> 2.5.0, and one more match for 2.7.1 --> 2.8.0. 

Here's my `_close` function to handle comparisons between tensors `a` and `b`:

```python
def _close(a, b, default=False):
    gtype = a.dtype
    
    if gtype in [torch.uint8, torch.int32, torch.int64]:
        if a.shape == b.shape: return torch.equal(a,b)
        return False

    if not default:
        if gtype == torch.float32:
            atol, rtol = 1e-6, 1e-5
        elif gtype == torch.bfloat16:
            atol, rtol = 1e-3, 1e-2
        else:
            atol, rtol = 1e-4, 1e-3
    else:
        atol, rtol = 1e-8, 1e-5
    return torch.allclose(a, b, rtol=rtol, atol=atol)
```

## Root Cause For Index Artifact Difference Between Consecutive PyTorch Versions

There were three consecutive PyTorch versions which broke index artifact reproducibility in `colbert-ai`. Listed below are the tensors that failed `torch.equal` (for integers) or `torch.allclose` (with bitsandbytes' tolerances):

- 2.0.1 --> 2.1.0
    - `ivf.pid.pt` (`ivf`: unique passage IDs (pids) per centroid ID, `ivf_lengths`: number of pids per centroid id)
    - `codes.pt` (centroid ID mapped to doc token IDs)
    - `residuals.pt` (distance between centroids and doc token embeddings)
    - `centroids.pt` (centroids of clustered sample doc token embeddings `local_sample_embs`)
    - `bucket_cutoffs` (the quantization bins)
- 2.4.1 --> 2.5.0
    - `ivf.pid.pt` (`ivf` and `ivf_lengths`)
    - `codes.pt`
    - `residuals.pt` 
    - `centroids.pt`
    - `bucket_cutoffs`
- 2.7.1 --> 2.8.0
    - `residuals.pt`

In the following sections I'll detail the root cause for index artifact divergence.

### 2.0.1 --> 2.1.0: `BertModel` Forward Pass for Any `input_ids`

The first critical intermediate indexing tensor created is [`local_sample_embs`](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/collection_indexer.py#L137). This is a sample of document token embeddings used to calculate centroids. The sample passages are passed to `Checkpoint.docFromText`, which calls `Checkpoint.doc`, which ultimately calls `Checkpoint.bert`. 

`sample_pids`, the sample of passage IDs selected for encoding, were identical between 2.0.1 and 2.1.0, but `local_sample_embs` did not pass `torch.allclose` (with bnb tolerances). This was the smell that led me to compare the `Checkpoint.bert` model layer outputs between PyTorch versions using `register_forward_hook`. I tried a variety of input tokens (different batches of passages, random text, single letter strings) and in all cases, model layer outputs between PyTorch versions failed `torch.allclose`. I thus concluded that something in PyTorch changed between 2.0.1 and 2.1.0 to cause this. You can read more details of this exploration in [another blog post](https://vishalbakshi.github.io/blog/posts/2025-09-13-colbert-maintenance/).

To confirm that the `local_sample_embs` divergence caused the divergence in downstream index artifacts, I replaced the `local_sample_embs` in the `torch==2.1.0` install with `local_sample_embs` from the `torch==2.0.1` install and the final index artifacts passed `torch.allclose`. Interestingly, even though all final index artifacts were similar, the intermediate `codes.pt` (centroid ID mapped to doc token IDs) was not. I did a deep dive in [a separate blog post](https://vishalbakshi.github.io/blog/posts/2025-09-09-colbert-maintenance/) where I discovered that using `Tensor.sort` results in different sort indices in `torch==2.0.1` and `torch==2.1.0`.

### 2.4.1 --> 2.5.0: `BertModel` Forward Pass for Some Batch Sizes

I saw a similar result when changing the `colbert-ai` PyTorch version from 2.4.1 to 2.5.0: identical `sample_pids`, diverging `local_sample_embs`. In this case, however, not all `input_ids` caused a divergence between PyTorch versions. Specifically, inputs of the following batch sizes resulted in model layer outputs passing `torch.allclose`: 71, 72, 70, 73, 68, 66, 115, 64, 63, 62, 61, 67, 69. And the following batches _failed_ `torch.allclose`: 79, 78, 77, 194, 82, 80, 90, 86, and 83. I concluded that something in PyTorch changed between 2.4.1 and 2.5.0 which made the `BertModel` forward pass have _batch variance_. Interestingly, it was at this time that I read the excellent Thinking Machines' [blog post about LLM non-determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#:~:text=As%20it%20turns%20out%2C%20our%20request%E2%80%99s%20output%20does%20depend%20on%20the%20parallel%20user%20requests.%20Not%20because%20we%E2%80%99re%20somehow%20leaking%20information%20across%20batches%20%E2%80%94%20instead%2C%20it%E2%80%99s%20because%20our%20forward%20pass%20lacks%20%E2%80%9Cbatch%20invariance%E2%80%9D%2C%20causing%20our%20request%E2%80%99s%20output%20to%20depend%20on%20the%20batch%20size%20of%20our%20forward%20pass).

### 2.7.1 --> 2.8.0: Difference in `torch.nn.functional.normalize` Output

When comparing the 2.7.1 and 2.8.0 index artifacts, all artifacts but `residuals.pt` passed `torch.allclose` with bnb tolerances. `residuals.pt` are the difference between the document token embeddings and the centroids](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/codecs/residual.py#L176):

```python
residuals_ = batch - centroids_
```

`batch` not only passes `torch.allclose` between PyTorch versions, but also passes `torch.equal`. Whereas `centroids_` only passes `torch.allclose`. Looking deeper at how `centroids_` are calculated, they are [normalized and then stored in half precision](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/collection_indexer.py#L306-L308). The pre-norm centroids pass `torch.equal` between PyTorch versions but the post-norm centroids do not. Additionally, testing this on random values, the pre-norm tensors are equal between PyTorch versions but the post-norm tensors are not. You can read more details on this in [another blog post](https://vishalbakshi.github.io/blog/posts/2025-09-14-colbert-maintenance/).

## Conclusion

In all three cases, when changing PyTorch versions, `colbert-ai` indexing functionality does not break, but reproducibility does. To recap the root causes:

- `torch==2.0.1` --> `torch==2.1.0`: `BertModel` forward pass outputs diverge **for any inputs** + `Tensor.sort` indices order changes.
- `torch==2.4.1` --> `torch==2.5.0`: `BertModel` forward pass outputs diverge **depending on batch size**.
- `torch==2.7.1` --> `torch==2.8.0`: `torch.nn.functional.normalize` outputs diverge.

I don't think these root causes can be addressed in the `colbert-ai` codebase as they seem to be purely PyTorch changes. However, I'm documenting them here (and will link this blog post in the next `colbert-ai` release notes) as users will experience index artifact changes when using different PyTorch versions.

Next up: comparing and documenting search and training artifacts across PyTorch versions.


## Appendix

In this section I'll detail final and intermediate index tensor artifact comparisons between PyTorch versions where `torch.allclose` was `False` using default tolerances. I'll also document integer tensor artifacts separately with `torch.equal` for tensors (which I was embarrassingly until now comparing with `torch.allclose`, 🤦) and `==` for non-tensors.  

### `torch==2.0.1` vs `torch==2.1.0`

#### Final Index Artifacts 

Using the more lenient bitsandbytes tolerances, <mark>`avg_residual.pt` and `bucket_weights.pt` pass `torch.allclose`</mark> while `bucket_cutoffs` and `centroids` do not.

###### Integer Tensors

|Artifact|Description|dtype|`torch.equal`|
|:-:|:-:|:-:|:-:|
|`codes.pt`|centroid id mapped to doc token embeddings|`torch.int32`|`False`|
|`residuals.pt`|difference between centroid and doc token embeddings|`torch.uint8`|`False`|
|`ivf.pid.pt` (ivf)|unique pids per centroid id|`torch.int32`|<mark>shape mismatch</mark>|
|`ivf.pid.pt` (ivf_lengths)|number of pids per centroid id|`torch.int64`|`False`

##### Float Tensors

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`avg_residual.pt`|Average difference between centroids and doc token embeddings|`torch.float16`|`False`|`True`|
|`buckets.pt` (`bucket_cutoffs`)|The quantization bins|`torch.float32`|`False`|`False`|
|`buckets.pt` (`bucket_weights`)|The quantization values for each bin|`torch.float16`|`False`|`True`|
|`centroids.pt`|Centroids of clustered sample doc token embeddings|`torch.float16`|`False`|`False`|

#### Intermediate Index Artifacts

"Intermediate" artifacts are tensors saved in the middle of the indexing pipeline by adding `torch.save` calls in `/colbert/indexing/collection_indexer.py` or `/colbert/modeling/checkpoint.py`.

##### Integer Tensors

Some of the intermediate artifacts are not tensors so the equality column I'm titling "Equal" instead of `torch.equal`.

|Artifact|Description|dtype|Equal|
|:-:|:-:|:-:|:-:|
|`sample_pids.pt`|A sample of passage ids used to calculate centroids|`int`|`True`|
|`num_passages.pt`|Number of sampled passages|`int`|`True`|
|`doclens.pt`|List of number of tokens per document|`int`|`True`|

##### Float Tensors

Using the more lenient bitsandbytes tolerances, none of the `torch.allclose` calls pass.

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`local_sample_embs.pt`|Embeddings of sample document passages used to calculate centroids|`torch.float16`|`False`|`False`
|`sample.pt`|95% of the values from `local_sample_embs.half()`|`torch.float16`|`False`|`False`
|`sample_heldout.pt`|5% of the values from `local_sample_embs.half()`|`torch.float16`|`False`|`False`
|`batches.pt`|1 batch of encoded passages|`torch.float16`|`False`|`False`
|`D.pt`|sorted and reshaped `batches`|`torch.float16`|`False`|`False`

## Core Difference: `BertModel` Forward Pass

Swapping the `local_sample_embs.pt` and `embs_{chunk_idx}.pt` tensors in the `torch==2.1.0` ColBERT install with the ones generated in the `torch==2.0.1` install resolves all final index artifacts discrepancies, even when using default tolerances. This led me to uncover that the core difference between 2.0.1 and 2.1.0 is the `BertModel` forward pass. The intermediate and final `BertModel` layer outputs all fail `torch.allclose` (for both sets of tolerances), no matter what the input tokens are (I tried different batch sizes and also a single letter `"a"`).

::: {.callout-note}
## What does "Swapping" Mean?
"Swapping" means loading the tensor right before it's saved:

```python
if SWAP == 'True': local_sample_embs = torch.load(f"{SWAP_ROOT}/local_sample_embs.pt") # ADDED BY VISHAL
torch.save(local_sample_embs, f"{ROOT}/local_sample_embs.pt") # ADDED BY VISHAL
torch.save(local_sample_embs.half(), os.path.join(self.config.index_path_, f'sample.{self.rank}.pt'))
```

```python
if SWAP == 'True': embs = torch.load(f"{SWAP_ROOT}/embs_{chunk_idx}.pt") # ADDED BY VISHAL
torch.save(embs, f"{ROOT}/embs_{chunk_idx}.pt") # ADDED BY VISHAL
torch.save(doclens, f"{ROOT}/doclens.pt") # ADDED BY VISHAL
self.saver.save_chunk(chunk_idx, offset, embs, doclens) # offset = first passage index in chunk
```
:::

## Peculiar Finding: Different Intermediate `codes` Artifact Yields Identical Final `ivf.pid.pt` Artifact

Even after swapping `local_sample_embs.pt` and `embs`, the intermediate `codes` (not shown in table above) between PyTorch versions did not pass `torch.allclose` (even with the more lenient bitsandbytes tolerances). 

## `torch==2.4.1` vs `torch==2.5.0`

### Final Index Artifacts

#### Integer Tensors

|Artifact|Description|dtype|`torch.equal`|
|:-:|:-:|:-:|:-:|
|`codes.pt`|centroid id mapped to doc token embeddings|`torch.int32`|`False`|
|`residuals.pt`|difference between centroid and doc token embeddings|`torch.uint8`|`False`|
|`ivf.pid.pt` (ivf)|unique pids per centroid id|`torch.int32`|<mark>shapes mismatch</mark>|
|`ivf.pid.pt` (ivf_lengths)|number of pids per centroid id|`torch.int64`|`False`

#### Float Tensors

With bnb tolerances, `avg_residual.pt` and `bucket_weights` pass `torch.allclose` between PyTorch versions.

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`avg_residual.pt`|Average difference between centroids and doc token embeddings|`torch.float16`|`False`|`True`|
|`buckets.pt` (`bucket_cutoffs`)|The quantization bins|`torch.float32`|`False`|`False`|
|`buckets.pt` (`bucket_weights`)|The quantization values for each bin|`torch.float16`|`False`|`True`|
|`centroids.pt`|Centroids of clustered sample doc token embeddings|`torch.float16`|`False`|`False`|

### Intermediate Index Artifacts

#### Integer Tensors

|Artifact|Description|dtype|Equal|
|:-:|:-:|:-:|:-:|
|`sample_pids.pt`|A sample of passage ids used to calculate centroids|`int`|`True`|
|`num_passages.pt`|Number of sampled passages|`int`|`True`|
|`doclens.pt`|List of number of tokens per document|`int`|`True`|

#### Float Tensors

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`local_sample_embs.pt`|Embeddings of sample document passages used to calculate centroids|`torch.float16`|`False`|`False`
|`sample.pt`|95% of the values from `local_sample_embs.half()`|`torch.float16`|`False`|`False`
|`sample_heldout.pt`|5% of the values from `local_sample_embs.half()`|`torch.float16`|`False`|`False`
|`batches.pt`|1 batch of encoded passages|`torch.float16`|`True`|`True`
|`D.pt`|sorted and reshaped `batches`|`torch.float16`|`True`|`True`


::: {.callout-important}
`batches.pt` did not pass `torch.allclose` for a 1000-document subset as the final batch item had 8 items and .
:::

## Core Difference: Something in `BertModel` 

Swapping the `local_sample_embs.pt` and `embs_{chunk_idx}.pt` tensors in the `torch==2.5.0` ColBERT install with the ones generated in the `torch==2.4.1` install resolves all final _and intermediate_ index artifacts discrepancies, even when using the smaller default tolerances. However, it's unclear what is causing the divergence in the `BertModel`. 

When sampling and embedding just the first 1000 passages (with `checkpoint.bert`), [the `BertModel` intermediate dense layer outputs different tensors between PyTorch versions 2.4.1 and 2.5.0 when using mixed precision (for small batch sizes)](<https://vishalbakshi.github.io/blog/posts/2025-08-26-colbert-maintenance/#:~:text=Mixed%20precision%20(,between%20PyTorch%20versions.>) <mark>this divergence also seems to be related to the number of tokens</mark>. When embedding the full dataset (69_199 passages), the third batch of 1600 passages caused a divergence in `BertModel` layer outputs. 

::: {.callout-note collapse="true"}
## What does "Swapping" Mean?
"Swapping" means loading the tensor right before it's saved:

```python
if SWAP == 'True': local_sample_embs = torch.load(f"{SWAP_ROOT}/local_sample_embs.pt") # ADDED BY VISHAL
torch.save(local_sample_embs, f"{ROOT}/local_sample_embs.pt") # ADDED BY VISHAL
torch.save(local_sample_embs.half(), os.path.join(self.config.index_path_, f'sample.{self.rank}.pt'))
```

```python
if SWAP == 'True': embs = torch.load(f"{SWAP_ROOT}/embs_{chunk_idx}.pt") # ADDED BY VISHAL
torch.save(embs, f"{ROOT}/embs_{chunk_idx}.pt") # ADDED BY VISHAL
torch.save(doclens, f"{ROOT}/doclens.pt") # ADDED BY VISHAL
self.saver.save_chunk(chunk_idx, offset, embs, doclens) # offset = first passage index in chunk
```
:::

## `torch==2.7.1` vs `torch==2.8.0`

Using the more lenient bitsandbytes tolerances, ALL `torch.allclose` calls pass. <mark>It's interesting to note that while `centroids.pt` (floats) passes `torch.allclose`, `residuals.pt` (integers) is not equal across PyTorch versions.</mark>

### Final Index Artifacts

#### Integer Tensors

|Artifact|Description|dtype|`torch.equal`|
|:-:|:-:|:-:|:-:|
|`codes.pt`|centroid id mapped to doc token embeddings|`torch.int32`|`True`|
|`residuals.pt`|difference between centroid and doc token embeddings|`torch.uint8`|`False`|
|`ivf.pid.pt` (ivf)|unique pids per centroid id|`torch.int32`|`True`|
|`ivf.pid.pt` (ivf_lengths)|number of pids per centroid id|`torch.int64`|`True`

#### Float Tensors

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`avg_residual.pt`|Average difference between centroids and doc token embeddings|`torch.float16`|`True`|`True`|
|`buckets.pt` (`bucket_cutoffs`)|The quantization bins|`torch.float32`|`True`|`True`|
|`buckets.pt` (`bucket_weights`)|The quantization values for each bin|`torch.float16`|`True`|`True`|
|`centroids.pt`|Centroids of clustered sample doc token embeddings|`torch.float16`|`False`|`True`|

::: {.callout-important}
When using default tolerances, [the normalized half-precision centroids cause the floating-point error](https://vishalbakshi.github.io/blog/posts/2025-09-02-colbert-maintenance/#inspecting-.half-behavior:~:text=The%20half%20precision%20random%20tensors%20(before%20normalization)%20are%20identical%20between%20torch%20versions%20but%20the%20half%20precision%20normalized%20tensors%20are%20not.). 
:::

### Intermediate Index Artifacts

All of my intermediate index artifacts pass `torch.allclose` regardless of which tolerances are used. 

#### Integer Tensors

|Artifact|Description|dtype|Equal|
|:-:|:-:|:-:|:-:|
|`sample_pids.pt`|A sample of passage ids used to calculate centroids|`int`|`True`|
|`num_passages.pt`|Number of sampled passages|`int`|`True`|
|`doclens.pt`|List of number of tokens per document|`int`|`True`|

#### Float Tensors

|Artifact|Description|dtype|Default|bnb
|:-:|:-:|:-:|:-:|:-:|
|`local_sample_embs.pt`|Embeddings of sample document passages used to calculate centroids|`torch.float16`|`True`|`True`
|`sample.pt`|95% of the values from `local_sample_embs.half()`|`torch.float16`|`True`|`True`
|`sample_heldout.pt`|5% of the values from `local_sample_embs.half()`|`torch.float16`|`True`|`True`
|`batches.pt`|1 batch of encoded passages|`torch.float16`|`True`|`True`
|`D.pt`|sorted and reshaped `batches`|`torch.float16`|`True`|`True`


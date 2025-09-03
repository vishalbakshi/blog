---
title: PyTorch Version Impact on ColBERT Index Artifacts&#58; 2.7.1 --> 2.8.0
date: "2025-09-02"
author: Vishal Bakshi
description: Analysis of ColBERT indexing differences between PyTorch 2.7.1 and 2.8.0 shows the root cause is half precision divergence in normalized centroids.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

In a [previous blog post](https://vishalbakshi.github.io/blog/posts/2025-08-18-colbert-maintenance/) I showed how I traced index artifact differences between `colbert-ai` installs using `torch==1.13.1` (the current pinned version) and `torch==2.1.0` (the first version which produces different index artifacts) to a difference in floating point differences in the forward pass of the underlying `BertModel`. 

In a [subsequent blog post](https://vishalbakshi.github.io/blog/posts/2025-08-26-colbert-maintenance/) I showed how the index artifact differences between `colbert-ai` installs using `torch==2.4.1` and `torch==2.5.0` (the next two versions with differences) was due to floating point divergence in BERT’s intermediate linear layer under mixed precision with small batch sizes.

In this blog post, I'll show how the index artifacts differencces between `torch==2.7.1` and `torch==2.8.0` is due to floating point differences between half precision normalized centroid tensors.

## Index Artifact Comparison

There are two index artifacts that are different between `colbert-ai` installs using `torch==2.7.1` and `torch==2.8.0`: centroids.pt and the related residuals.pt (the difference between document token embeddings and centroids). This divergence does <mark>NOT</mark> result in a divergence in the critical `ivf.pt` (document token IDs) and `values` (centroid IDs) tensors. In other words, the most important mapping from document token IDs to centroid IDs does not change even though centroids floating point values change enough to fail `torch.allclose`.


|Artifact|`torch.allclose`|
|:-:|:-:|
|`sampled_pids`|`True`
|`num_passages`|`True`
|`local_sample_embs`|`True`
|<mark>`centroids`</mark>|<mark>`False`</mark>
|`bucket_cutoffs`|`True`
|`bucket_weights`|`True`
|`avg_residual`|`True`
|<mark>`residuals`</mark>|<mark>`False`</mark>
|`sample`|`True`
|`sample_heldout`|`True`
|`embs`|`True`
|`doclens`|`True`
|`codes`|`True`
|`ivf`|`True`
|`values`|`True`
|`tensorize_output`|`True`
|`batches`|`True`
|`D`|`True`

## Inspecting `centroids.pt`

I added the following `torch.save` calls inside [`CollectionIndexer._train_kmeans`](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/collection_indexer.py#L280):

```python
if do_fork_for_faiss:
    ...
else:
    args_ = args_ + [[[sample]]]
    centroids = compute_faiss_kmeans(*args_)
    torch.save(centroids, f"{ROOT}/prenorm_centroids.pt") # ADDED BY VISHAL
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    torch.save(centroids, f"{ROOT}/postnorm_centroids.pt") # ADDED BY VISHAL

if self.use_gpu:
    centroids = centroids.half()
    torch.save(centroids, f"{ROOT}/half_centroids.pt") # ADDED BY VISHAL
else:
    centroids = centroids.float()
```

I then compared `prenorm_centroids.pt`, `postnorm_centroids.pt` and `half_centroids.pt` between both `colbert-ai` installs using `torch.allclose`:

```python
prenorm_centroids.pt torch.allclose:    True
prenorm_centroids.pt MAD: 0.0   True

postnorm_centroids.pt torch.allclose:   True
postnorm_centroids.pt MAD: 7.014875902378037e-10        False

half_centroids.pt torch.allclose:       False
half_centroids.pt MAD: 9.313225746154785e-10    False
```

The pre-normalization and post-normalization centroids are identical across torch versions, but the <mark>half precision normalized centroids</mark> diverge.

## Inspecting `.half` Behavior

Are all half precision tensors different across torch versions? No. There are a number of index artifacts that are converted to half precision during indexing and are identical between torch versions: `avg_residual.pt`, `D.pt`, `bucket_weights.pt`, and `embs.pt`.

Furthermore, I created a random tensor, its half precision version, and its normalized version (full and half precision)...

```python
torch.manual_seed(13)
t = torch.empty(1024, 96).uniform_(-0.4, 0.4)
torch.save(t, "t.pt")
torch.save(t.half(), "half_t.pt")

t = torch.nn.functional.normalize(t, dim=-1)
torch.save(t, "norm.pt")
torch.save(t.half(), "half_norm.pt")
```

...and compared it between torch versions:

|Artifact|`torch.allclose`|
|:-:|:-:|
|`t.pt`|`True`|
|`half_t.pt`|`True`|
|`norm.pt`|`True`|
|<mark>`half_norm.pt`</mark>|<mark>`False`</mark>|

The half precision random tensors (before normalization) are identical between torch versions but the half precision normalized tensors are not. <mark>It was not apparent from a cursory review of the [PyTorch Release 2.8.0 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.8.0) what caused this behavior.</mark> Sonnet 4 is confident it's due to PyTorch PR [#153888](https://github.com/pytorch/pytorch/pull/153888) (upgrade cuDNN frontend submodule to 1.12) but that could just be a shot in the dark and I can't verify it.

## Next Steps

I have now identified what causes index artifacts to diverge between the three pairs of PyTorch versions in question (1.13.1 --> 2.1.0, 2.4.1 --> 2.5.0, and 2.7.1 --> 2.8.0). Next I will inspect search related artifacts and understand where there are differences and why. Once that's complete, I'll look into training artifacts. Finally, I'll test index, search and training for different Python versions (3.9, 3.10, 3.11, 3.12, and 3.13). Unless something else emerges in my analysis, after Python version testing is complete, I'll be able to push the next release of `colbert-ai` with the dependency change from `"torch==1.13.1"` to (most likely) `"torch>=1.13.1,<=2.8.0"`.
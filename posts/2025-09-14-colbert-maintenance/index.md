---
title: Debugging ColBERT Index Differences Between PyTorch 2.7.1 and 2.8.0
date: "2025-09-14"
author: Vishal Bakshi
description: ColBERT index artifacts created with PyTorch 2.8.0 fail numerical equality tests compared to those created with 2.7.1 (even with more lenient `torch.allclose` tolerances). Through systematic debugging of intermediate tensors, I traced the root cause to precision changes in PyTorch's vector normalization implementation.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

I've been redoing my `colbert-ai` index comparisons between PyTorch versions using [bitsandbytes' `torch.allclose` tolerances](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/39dd8471c1c0677001d0d20ba2218b14bf18fd00/tests/test_optim.py#L189-L194). There are three PyTorch version changes that cause index artifact changes: 2.0.1 to 2.1.0 ([`BertModel` forward pass outputs diverge for all inputs](https://vishalbakshi.github.io/blog/posts/2025-09-13-colbert-maintenance/)), 2.4.1 to 2.5.0 ([certain batch sizes cause `BertModel` output divergence](https://vishalbakshi.github.io/blog/posts/2025-09-11-colbert-maintenance/)), and 2.7.1 to 2.8.0 (detailed in this blog post).

## Difference Between PyTorch Versions: `residuals.pt`

When using the bitsandbytes' `torch.allclose` tolerances, all final index artifacts pass `torch.allclose` except `residuals.pt`. Residuals are a key component in the indexing pipeline, they are the distance between document token embeddings and centroids. From [residual.py's `ResidualCodec.compress`](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/codecs/residual.py#L167):

```python
def compress(self, embs, chunk_idx): # chunk_idx ADDED BY VISHAL
    codes, residuals = [], []

    for batch in embs.split(1 << 18):
        if self.use_gpu:
            batch = batch.cuda().half()
        codes_ = self.compress_into_codes(batch, out_device=batch.device)
        centroids_ = self.lookup_centroids(codes_, out_device=batch.device)

        residuals_ = (batch - centroids_)
        torch.save(residuals_, f"{ROOT}/residuals__{chunk_idx}.pt") # ADDED BY VISHAL
        torch.save(codes_, f"{ROOT}/codes__{chunk_idx}.pt") # ADDED BY VISHAL
        torch.save(batch, f"{ROOT}/batch_{chunk_idx}.pt") # ADDED BY VISHAL
        torch.save(centroids_, f"{ROOT}/centroids__{chunk_idx}.pt") # ADDED BY VISHAL
        codes.append(codes_.cpu())
        residuals.append(self.binarize(residuals_).cpu())

    codes = torch.cat(codes)
    torch.save(codes, f"{ROOT}/compress_codes_{chunk_idx}.pt")
    residuals = torch.cat(residuals)

    return ResidualCodec.Embeddings(codes, residuals)
```

The key line is:

```python
residuals_ = (batch - centroids_)
```

As you can see above, I have added `torch.save` calls to compare those intermediate index artifacts between PyTorch versions.

I figured that since `residuals_` do not pass `torch.allclose` between torch versions, `batch` and `centroids_` must not as well. I was wrong! `batch` does not only pass `torch.allclose` but also passes `torch.equal` between torch versions. `centroids_` passes `torch.allclose` but not `torch.equal`. Even though `centroids_` values are within floating-point tolerance (`torch.allclose` passes), the small differences get amplified during the subtraction operation that creates `residuals_`. This amplification pushes the final result outside the tolerance bounds, causing `residuals_` to fail `torch.allclose`.

## Difference Between PyTorch Versions: `centroids.pt`

This begs the question: why are `centroids_` between PyTorch versions not exactly equal? In other words, why don't `centroids_` pass `torch.equal` like `batch` does? To figure this out, I added `torch.save` calls to [`_train_kmeans`](https://github.com/stanford-futuredata/ColBERT/blob/501c29d9e0b7f7b393e36c4177ec2b141a253114/colbert/indexing/collection_indexer.py#L280) where the centroids are created:

```python
def _train_kmeans(self, sample, shared_lists):
        centroids = compute_faiss_kmeans(*args_)
    torch.save(centroids, f"{ROOT}/prenorm_centroids.pt") # ADDED BY VISHAL
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    if POSTNORM_CENTROIDS_SWAP == "True": centroids = torch.load(f"{POSTNORM_CENTROIDS_SWAP_ROOT}/postnorm_centroids.pt") # ADDED BY VISHAL
    torch.save(centroids, f"{ROOT}/postnorm_centroids.pt") # ADDED BY VISHAL
    if self.use_gpu:
        centroids = centroids.half()
        torch.save(centroids, f"{ROOT}/half_centroids.pt") # ADDED BY VISHAL
    else:
        centroids = centroids.float()

    return centroids
```

There are three versions of centroids I save: `prenorm_centroids.pt` (the output of `compute_faiss_kmeans`), `postnorm_centroids.pt` (the output of `torch.nn.functional.normalize(centroids, dim=-1)`) and `half_centroids.pt` (the output of `centroids.half()`).

I compare each tensor (created with `torch==2.7.1` and `torch==2.8.0`) with both `torch.allclose` and `torch.equal`:

|Tensor|`torch.allclose`|`torch.equal`|
|:-:|:-:|:-:|
|prenorm_centroids.pt|`True`|`True`|
|postnorm_centroids.pt|`True`|`False`
|half_centroids.pt|`True`|`False`|

The pre-norm centroids are _exactly the same_ between PyTorch versions, but the post-norm centroids are not. To confirm that the divergence between PyTorch versions is the normalization operation, I replace the 2.8.0 `postnorm_centroids.pt` with the 2.7.1 ones (the `if POSTNORM_CENTROIDS_SWAP == "True"` line in the code above) and all final and intermediate index artifacts (including `residuals.pt`) pass `torch.allclose` between PyTorch versions.

To confirm that there exists a difference in normalization between PyTorch versions 2.7.1 and 2.8.0 I generate the following tensors with each install:

```python
torch.manual_seed(13)
t = torch.empty(1024, 96).uniform_(-0.4, 0.4)
torch.save(t, f"{MOUNT}/{project}/{date}-{source}-{nranks}/t.pt")
torch.save(t.half(), f"{MOUNT}/{project}/{date}-{source}-{nranks}/half_t.pt")

t = torch.nn.functional.normalize(t, dim=-1)
torch.save(t, f"{MOUNT}/{project}/{date}-{source}-{nranks}/norm.pt")
torch.save(t.half(), f"{MOUNT}/{project}/{date}-{source}-{nranks}/half_norm.pt")
```

Comparing the four tensors (`t.pt`, `half_t.pt`, `norm.pt` and `half_norm.pt`) between PyTorch versions:

|Tensor|`torch.allclose`|`torch.equal`|
|:-:|:-:|:-:|
|t.pt|`True`|`True`|
|half_t.pt|`True`|`True`|
|norm.pt|`True`|`False`|
|half_norm.pt|`True`|`False`|

While all tensors pass `torch.allclose` (bnb tolerances) the normalized tensors (full precision and half precision) fail `torch.equal` between PyTorch versions. When used in further operations (as `centroids_` are when calculating `residuals_ = batch - centroids_`) this inequality compounds and amplifies floating point differences enough to fail `torch.allclose` for the residuals.

## Closing Thoughts

When working with floating point values, it's easy to dismiss minor differences. The recent [Thinking Machines' blog post](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) communicated this sentiment:

> What’s wrong with bumping up the atol/rtol on the failing unit test?

As I've been exploring `colbert-ai` index artifact differences across PyTorch versions, it's been tempting to consider that "fix". However, by caring about failed `torch.allclose` or `torch.equal` I've learned a lot about how small differences impact index artifacts downstream, and have gained a better understanding of how changes in PyTorch can impact `colbert-ai`. While I may not cover all such impacts, I'm hoping that documenting them here will help some engineer somewhere who is debugging why their RAG pipeline has subtle changes after bumping up PyTorch versions.

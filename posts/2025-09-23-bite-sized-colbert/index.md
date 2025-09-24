---
title: Bite-Sized `colbert-ai` &#58; Building A Maintainer's Mental Model
date: "2025-09-23"
author: Vishal Bakshi
description: A set of concise explanations of the `colbert-ai` mental model I keep in mind while doing maintenance work. 
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

I recently published a blog post on how I'm [excited to start my professional ML career](https://vishalbakshi.github.io/blog/posts/2025-09-18-career/) after 8 years of being a data analyst, educator, fast.ai student and community member, and now stanford-futuredata/ColBERT maintainer. In response, I have received messages from folks in the industry wanting to talk about career/project opportunities. As I was preparing for these conversations, I realized that I don't have bite-size explanations of my `colbert-ai` maintenance work mental model. This blog post attempts to articulate a first draft of that mental model. My goal is that each section in this post has a reading time of 2 minutes or less, with a full reading time of 15 minutes or less. While I have more verbose, exploratory, detailed-focused blog posts on the [search pipeline](https://vishalbakshi.github.io/blog/posts/2024-12-24-PLAID-ColBERTv2-scoring-pipeline/) or the [indexing pipeline](https://vishalbakshi.github.io/blog/posts/2025-03-12-RAGatouille-ColBERT-Indexing-Deep-Dive/), it's hard to apply those writings to short, introductory conversations, which are what these "inbound" calls are.

::: {.callout-note}
Looking to Chat
If you’re working on something exciting in ML and think I could contribute as an employee or consultant, I’d love to chat: vdbakshi at gmail dot com.
:::

## `colbert-ai` As a Product

I quickly realized that maintaining `colbert-ai` is like maintaining a product. The end users are ML engineers and researchers, and the fundamental user experience is reproducibility. When PyTorch version changes break reproducibility, I have to systematically figure out why and whether it can be fixed. If it can't be fix, I have to document what I found so users have at least a starting point to their troubleshooting.

## My Approach to Maintaining `colbert-ai`

1. **Understand the indexing, searching and training pipelines.** I need to know where artifacts are generated and what they mean, so I can correctly interpret any reproducibility changes. This foundational work took weeks back in January/February.

2. **Handle different data types properly**. Integer tensors are straightforward to compare. But floating-point tensors require `torch.allclose` with tolerance settings. 

3. **Execute systematic comparisons.** I add `torch.save` calls throughout the pipeline and use separate Docker images for each PyTorch version. After running the pipeline, I load each artifact pair and apply my tolerance heuristics to identify where reproducibility breaks.

4. **Isolate the root cause.** I dive into the specific code generating non-reproducible artifacts and isolate the PyTorch calls. To confirm my analysis, I swap the diverging artifact between PyTorchversions—if that fixes reproducibility, I know I found the root cause. 

I document everything in blog posts to practice communicating my approach for the eventual release notes.


## `colbert-ai`: The Three-Headed Monster

Any change to `colbert-ai` has the potential break reproducibility in three broad pipelines of functionality: indexing, search and training. I think of indexing as the "primary" pipeline so I always test it first, and search as the "secondary" pipeline (as search results depend on the index). I think of training separately from the index-search duo and I test it last. <mark>This might just be a reflection of my familiarity with index/search</mark>.

## The `colbert-ai` Indexing Pipeline

The end goal is to map centroid IDs to passage IDs so that we can lookup passage IDs close to (i.e. near the same centroids as) the query tokens. Centroids of what? The whole document collection's token embeddings? No, just a sample of token embeddings. This is why the first critical index artifact created (imo) is `local_sample_embs`, which are the BERT-encoded token embeddings of a sample of passages. Once centroids are calculated, we can process the whole document collection by finding the _difference_ between each token embeddings and its closest centroid (i.e. the _residuals_) and then storing a quantized version of the residual.

## The `colbert-ai` Search Pipeline

The end goal is to find those passages whose token embeddings are close to the query tokens. This happens in four stages, starting with a computationally inexpensive but large scale "rough draft" and ending with a more refined final pass on fewer documents:

- Stage 1: Given `q` query tokens and `n` number of nearest (i.e. largest cosine similarity) centroids per query, lookup all **passage IDs** close to those `q` x `n` centroids (this is why our final index folder needs a mapping from centroid ID to passage ID, `ivf.pid.pt`).
- Stage 2: Given the passage IDs from Stage 1, lookup the centroid IDs corresponding to the tokens in those passages (this is why our final index needs a mapping from centroid ID to token embedding ID, `codes.pt`). Filter out centroid IDs whose cosine similarity with any query token is less than some threshold. Add up cosine similarities of the remaining centroid ID across the query tokens to get one score per passage. Pick the top `ndocs` passages.
- Stage 3: Given the `ndocs` passage IDs from Stage 2, consider all centroid IDs, regardless of whether they passed the threshold, to recalculate the per-passage score. Pick the top `ndocs//4` passages.
- Stage 4: Only now do we decompress all token embeddings for the `ndocs//4` passages from Stage 3. Calculate the cosine similarity between each passage token and each query token and max-reduce to get one score per passage. Pick the top-`k` passages. These are your final results!

## The `colbert-ai` Training Pipeline

The goal is to train a model that scores relevant passages higher than irrelevant passages. `colbert-ai` does so by iterating over a batch of data that contains query, positive document, and negative document(s). The queries and passages are passed to the forward pass of the `ColBERT` model which generates scores (one per document). Using standard cross-entropy loss where the relevant document is positioned first in each group of candidate documents for a given query, the loss is calculated, and an optimizer step is performed. `colbert-ai` allows you to use in-batch negatives (i.e. using documents from other queries in the batch as negatives) and provide your own `target_scores` (i.e. distill scores from a teacher model using KL Divergence).

## Compressing Document Token Embeddings

The goal of compressing embeddings is two-fold: 1) enable the centroid-based search pipeline and 2) reduce storage space by using lower precision than 32-bit float32 or 16-bit float16. The compression (assuming 4-bit) involves:

- subtracting the (normalized) document token embeddings from the nearest (normalized) centroids to get your _residuals_.
- splitting the range of float values in _residuals_ into `2**n` (16) equal buckets (where `n=4` is the number of bits per value).
- create a mapping from bucket ID (for 4-bit there are 16 buckets from ID `0` to `15`) to original float values. This is where the lossiness of quantization comes into play.
- expanding that mapping tensor by adding a dimension of size `nbit` (4). So `[8,7, ...]` becomes `[[8, 8, 8, 8], [7, 7, 7, 7], ...]`.
- right-shifting those values by `torch.arange(nbits)`, so `[8,8,8,8]` becomes `[8,4,2,1]` (8 >> 0 = 8, 8 >> 1 = 4, 8 >> 2 = 2, 8 >> 3 = 1).
- perform a bitwise _and_ `&` between `1` and the right-shifted values, so `[8,4,2,1]` becomes `[0,0,0,1]`.
- combining the binarized bucket IDs into sequences of 8-bit integers (bucket ID `8` is `0001`, bucket ID `7` is `1110` and the combined value `00011110`, the 8-bit integer `32`).
- save the tensor of integers (combined 8-bit bucket IDs) as `residuals.pt`.

In the case of a 96-dimensional float32 residual embedding, you end up storing to disk a 48-dimensional int8 tensor. Note that to decompress you also need the integer centroid ID it was closest to.




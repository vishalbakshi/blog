
---
title: Analyzing PyTorch 2.0 Release Notes for ColBERT Dependency Impact
date: "2025-07-27"
author: Vishal Bakshi
description: In this blog post, I walk through the PyTorch 2.0 Release Notes items where I'm estimating there will be some kind of impact to ColBERT, which is currently dependent on torch==1.13.1.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background 

In this blog post, I walk through the [PyTorch 2.0 Release Note](https://github.com/pytorch/pytorch/releases/tag/v2.0.0) PRs where I'm estimating there will be some kind of impact to ColBERT as I update the torch dependency to 2.0 (ColBERT is currently dependent on torch==1.13.1). The level of detail in my analysis of a PyTorch PR is not necessarily signifying its importance. In some cases, I am using this analysis as an opportunity to get more familiar with details about the ColBERT codebase (such as the number of instances where `torch.cat` is used).

## Full Release Notes Analysis

You can find my item-by-item PyTorch 2.0 release notes analysis for ColBERT in [this Google Sheet](https://docs.google.com/spreadsheets/d/1sUEN7xo5-hLVoxF9NL_ibGxPaKlPzxmnMU46zf3wd-U/edit?usp=sharing). 

Overall, across 508 PyTorch PRs, I have estimated that 455 of them are not applicable to ColBERT and 42 (8.7%) have a potential impact. I was unclear if or how 11 of the PyTorch 2.0 PRs would affect ColBERT (2.6%).

There are 5 sections in the PyTorch 2.0 Release Notes, here's a break down of PRs by section that will have a potential (or unclear) impact on ColBERT:

|Section|# of PRs|
|:-:|:-:|
|Improvements|22
|Bug Fixes|21
|Performance|6
|Backwards Incompatible Changes|3
|Deprecations|1

In my estimation, the improvements and bug fixes PRs in PyTorch 2.0 will only improve the performance of ColBERT. That being said, there still may be noticeable differences in indexing, search, and training artifacts which may break tests I write for before/after comparisons. 

Fortunately, only two backward-compatible changes may affect ColBERT. 

There are 11 subsections in the PyTorch 2.0 Release Notes, here's a break down of PRs that will have a potential (or unclear) impact on ColBERT:

|Subsection|# of PRs|
|:-:|:-:|
|MPS|12
|Python API|8
|Cuda|8
|Releng|4
|Distributed|4
|Build|4
|ONNX|3
|Cpu|2
|Cpp API|2
|torch.nn API|1

I am including MPS-related PRs in this analysis just in case we consider making ColBERT compatible with MPS in the future.

I'll start by analyzing breaking changes, which are likely going to be the most impactful. 

## Backwards Incompatible Changes

### PR [#92731](https://github.com/pytorch/pytorch/pull/92731)

> Gradients are now set to None instead of zeros by default in `torch.optim.*.zero_grad()` and `torch.nn.Module.zero_grad()` (#92731)

There are two lines in ColBERT where`zero_grad` is called: in [colbert/utils/amp.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/amp.py#L37) and in [colbert/training/training.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/training/training.py#L61). I'm not sure how this change would affect ColBERT behavior, but flagging it as something to keep in mind.

### PR [92306](https://github.com/pytorch/pytorch/pull/92306)

> Algorithms `{Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, NAdam, RAdam, RMSProp, RProp, SGD}` default to faster `foreach` implementation when on CUDA + differentiable=`False`

This PR adds the following lines to `AdamW`, which is used in ColBERT's [`training.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/training/training.py#L60):

```python
if foreach is None:
    foreach = _default_to_foreach(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps],
        differentiable=differentiable)
```

`foreach` is `None` in ColBERT, as it's not specified and that's what it defaults to (`foreach: Optional[bool] = None`):

```python
optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
```

Since this is described as a "faster implementation", I would expect the training time to decrease. <mark>I'll be on the lookout for this when comparing training time benchmarks before/after upgrading to PyTorch 2.0</mark>.

### PR [#88913](https://github.com/pytorch/pytorch/pull/88913)

> Update `torch.tensor` and `nn.Parameter` to serialize all their attributes (#88913)

<mark>It's unclear what this PR is doing but since it's touching the `nn.Parameter` definition, I'm flagging it.</mark>

## Bug Fixes

I would expect PyTorch PRs that introduce bug fixes to only positively affect ColBERT. That being said, a positive effect is still a change and can potentially impact concrete artifacts during indexing, search and training. I am planning on curating a baseline set of these artifacts before I test the upgrade to PyTorch 2.0. 

### PR [#92810](https://github.com/pytorch/pytorch/pull/92810)

> Fix SIGSEGV on a big-endian machine when reading pickle data (#92810)

The PR states:

> This PR fixes SIGSEGV on a big-endian machine when reading pickle data.

I'm not familiar with the term "big-endian" so had to look it up:

> A big-endian system stores the most significant byte of a word at the smallest memory address and the least significant byte at the largest. A little-endian system, in contrast, stores the least-significant byte at the smallest address. ([source](https://en.wikipedia.org/wiki/Endianness))

Claude's understanding of the cpp method affected by this PR is that it affects the `torch.load` method. There are a number of ColBERT files that use `torch.load`:

- colbert/utils/coalesce.py uses it to [load `codes.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/coalesce.py#L47) (centroid id for each embedding in chunk) and  [load `residuals.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/coalesce.py#L66) (16-bits residual for each embedding in chunk).
- colbert/search/index_loader.py uses it to [load `ivf.pid.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_loader.py#L33) or [`ivf.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_loader.py#L36).
- colbert/utils/utils.py uses it [in `torch_load_dnn`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/utils.py#L44), [`load_checkpoint_raw`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/utils.py#L91) and [`load_ranking`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/utils.py#L205).
- colbert/indexing/index_manager.py uses it in [`load_index_part`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/index_manager.py#L20).
- colbert/indexing/codecs/residual_embeddings.py uses it in [`load_codes`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual_embeddings.py#L86) and [`load_residuals`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual_embeddings.py#L93).
- colbert/indexing/codecs/residual.py uses it in `load` to load [centroids](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L141), [`avg_residual`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L142), and [`bucket_cutoffs, bucket_weights`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L143).
- colbert/indexing/collection_indexer.py uses it in [`_concatenate_and_split_sample`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/collection_indexer.py#L256).
- colbert/index_updater.py uses it in `_load_disk_ivf` to load [`ivf.pid.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L281) or [`ivf.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L286), in [`load_chunk_codes`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L312), and in [`_load_chunk_residuals`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L316).
- colbert/tests/index_coalesce_test.py uses it to load [multi-file `codes.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/tests/index_coalesce_test.py#L57), [single-file `codes.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/tests/index_coalesce_test.py#L66), [multi-file `residuals.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/tests/index_coalesce_test.py#L83) and [single-file `residuals.pt`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/tests/index_coalesce_test.py#L92).

### PR [#92315](https://github.com/pytorch/pytorch/pull/92315)

> Fix NVML visible device parsing (#92315)

> CUDA_VISIBLE_DEVICES can contain either ordinals or UUIDs Extend the logic to be able to parse it by UUID

<mark>I don't think this would affect any artifacts created during indexing/searching/training but would make it easier for PyTorch to identify GPUs.</mark>

### PR [#93095](https://github.com/pytorch/pytorch/pull/93095)

This PR fixes an error in [#93006](https://github.com/pytorch/pytorch/issues/93006) when using `topk`, which is used in the following places in ColBERT:


- `get_cells` in [colbert/search/candidate_generation.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L17)
- `score_pids` in colbert/search/index_storage.py to [filter centroids by the threshold](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L127), [filter `pids` using pruned centroid scores](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L152) and [filter `pids` using full centroid scores](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L161)
- filter scores in `colbert_score_reduce` for the `"flipr"` interaction method: [link1 ](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L146), [link 2](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L150)

Claude recommended also consider uses of `max` and `argmax` to be potentially impacted:

-  `get_cells` if `ncells==1` in [colbert/search/candidate_generation.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L15)
- colbert/modeling/colbert.py in [`ColBERT.score`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L121)
- colbert/modeling/colbert.py in [`colbert_score_reduce`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L135)
- colbert/indexing/codecs/residual.py in `ResidualCodec.compress_into_codes` on [GPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L215) and [CPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L217)
- colbert/search/strided_tensor.py in [`StridedTensor.lookup](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L74)
- colbert/search/strided_tensor_core.py in [`StridedTensorCore.__init__`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor_core.py#L27)
- colbert/modeling/checkpoint.py in [`Checkpoint.score](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L215)
- colbert/indexing/utils.py in [`optimize_ivf`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/utils.py#L48)


<mark>I would assume that the only impact this PR would have on PyTorch is avoiding any errors during the use of `topk` (no such errors have been reported on in the open issues).</mark>

### PR [#85596](https://github.com/pytorch/pytorch/pull/85596)

> Fix: half reduction with multiple sub-iterators (#85596)

Fixes [cuda low-precision reductions on large tensors produce wrong results #74438](https://github.com/pytorch/pytorch/issues/74438): 

> Reductions with low precision inputs (half, bfloat16) that need sub-iterators accumulate directly in output and thus truncate intermediate results

This would fix any issues related to the use of `half` in the following ColBERT files:

- in `ResidualCodec` for [`centroids`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L27), [`avg_residual`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L35), [`bucket_weights`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L40), [saving `centroids`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L159), [compressing token embeddings](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L172), calculating cosine similarity between token embeddings and centroids [on GPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L215).
- colbert/search/candidate_generation.py in `CandidateGeneration.generate_candidates` for queries when [using the GPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L52).
- colbert/indexing/collection_indexer.py in `CollectionIndexer._sample_embeddings` when [saving `local_sample_embs`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/collection_indexer.py#L181), in `CollectionIndexer.train_kmeans` for [centroids on the GPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/collection_indexer.py#L308), and in `CollectionIndexer.index` when [saving token embeddings](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/collection_indexer.py#L370).
- colbert/modeling/colbert.py in `ColBERT.doc` for [document token embeddings on the GPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L106)

<mark>If this PR fix is relevant to the use of `half` in the above files I would expect there to be numeric differences in indexing/search artifacts</mark>.

### PR [#86492](https://github.com/pytorch/pytorch/pull/86492)

> Fixes a memory leak by making autocast cache global instead of thread-local (#86492)

This PR adds a PyTorch test which:

> Verifies that the autocast cache is global. This is done by mocking out cache clearing at the end of the forward pass, running forward+backward with an explicit call to autocast in the backward, and verifying that the weight only get cast to float16 once.

Claude's analysis: 

> This PyTorch enhancement directly benefits ColBERT. By making the autocast cache global, this PR provides a performance improvement when training ColBERT with mixed precision. It reduces redundant computations during the backward pass, leading to faster and more efficient training without changing the model's functionality.

ColBERT uses `torch.cuda.amp.autocast` in the following files:

- colbert/utils/amp.py in [`MixedPrecisionManager.context`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/amp.py#L15) which is used in colbert/training/training.py during [`train`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/training/training.py#L96), and in colbert/modeling/checkpoint.py in [`Checkpoint.query`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L87) and [`Checkpoint.doc`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L93) to calculate query and document token embeddings, respectively.
- colbert/distillation/scorer in [`Scorer._score_pairs`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/distillation/scorer.py#L48).

### PR [#88898](https://github.com/pytorch/pytorch/pull/88898)

Fixes PyTorch [#88873](https://github.com/pytorch/pytorch/issues/88873):

> torch\utils\cpp_extension.py should be fixed or ninja compile will fail.

Gemini's analysis: Because ColBERT uses the very feature this PR is fixing (torch.utils.cpp_extension.py), the change is directly relevant. This bug fix is important for any developer or user who needs to compile and run ColBERT on a Windows machine. It ensures that ColBERT's performance-critical custom CUDA code can be built correctly, preventing potential compilation errors.

ColBERT uses `torch.utils.cpp_extension` in the following files:

- colbert/modeling/colbert.py in `ColBERT.try_load_torch_extensions` to [load `segmented_lookup.cpp`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L12) on CPU.
- colbert/search/index_storage.py in `IndexScorer.try_load_torch_extensions` to load [filter_pids.cpp](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L38) and [decompress_residuals.cpp](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L51).
- colbert/search/strided_tensor.py in `StridedTensor.try_load_torch_extensions` to [load `segmented_lookup.cpp`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L26) on CPU.
- colbert/indexing/codecs/residual.py in `ResidualCode.try_load_torch_extensions` to load [decompress_residuals.cpp](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L103)



<mark>This might be related to ColBERT [#317](https://github.com/stanford-futuredata/ColBERT/issues/371)</mark>

### PR [#90149](https://github.com/pytorch/pytorch/pull/90149)

> Fix a static initialization order fiasco in c10d (#90149)

Gemini's analysis: Because ColBERT's multi-GPU functionality is built directly on the PyTorch library that this PR is fixing, this change is highly relevant. This is a crucial stability improvement that makes ColBERT's distributed training and inference more reliable by preventing potential crashes at startup.

<mark>If Gemini's analysis is correct, this will make ColBERT's multi-GPU functionality more reliable and might address related open issues.</mark>

### PRs [#86956](https://github.com/pytorch/pytorch/pull/86956), [#86958](https://github.com/pytorch/pytorch/pull/86958)

> Fix issues with non-contiguous Tensor handling (#86956, #86958)

<mark>These are both MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS</mark>


### PRs [#94119](https://github.com/pytorch/pytorch/pull/94119), [#86240](https://github.com/pytorch/pytorch/pull/86240), [#91520](https://github.com/pytorch/pytorch/pull/91520), [#94442](https://github.com/pytorch/pytorch/pull/94442), [#94386](https://github.com/pytorch/pytorch/pull/94386)

> Fix issues with ops implementation torch.median (#90326, #88807), torch.{std,var} correction argument (#91203), torch.index_select (#94117, #91064), torch.cumsum (#94119), torch.where (#86240), torch.nn.Embedding (#82809), torch.nn.Softplus (#88555), torch.nn.functional.pad (#89864), torch.max (#91520), padding functions (#91522), torch.nn.functional.upsample (#91669), pooling functions (#91519, #94348), torch.nn.{NLLLoss,SmoothL1Loss} (#94226), torch.nn.SoftPlus (#94256), torch.masked_fill (#94263), torch.fill_ (#94479), torch.median (#94489), torch.nonzero (#94442), torch.nn.BatchNorm (#94351), torch.{min,max} (#94386), torch.nn.GELU (#94529), torch.nn.LSTM (#94889), #95137),torch.nn.Conv2d(#95078),torch.nn.functional.bilinear(#94892),torch.copy\_ (#95272),torch.max_pool2d(#94963),torch.div (#95769)

ColBERT uses topk in the following files:

- `get_cells` in [colbert/search/candidate_generation.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L17)
- `score_pids` in colbert/search/index_storage.py to [filter centroids by the threshold](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L127), [filter `pids` using pruned centroid scores](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L152) and [filter `pids` using full centroid scores](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L161)
- filter scores in `colbert_score_reduce` for the `"flipr"` interaction method: [link1 ](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L146), [link 2](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L150)

ColBERT uses max/argmax in the following files:

-  `get_cells` if `ncells==1` in [colbert/search/candidate_generation.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L15)
- colbert/modeling/colbert.py in [`ColBERT.score`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L121)
- colbert/modeling/colbert.py in [`colbert_score_reduce`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L135)
- colbert/indexing/codecs/residual.py in `ResidualCodec.compress_into_codes` on [GPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L215) and [CPU](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L217)
- colbert/search/strided_tensor.py in [`StridedTensor.lookup](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L74)
- colbert/search/strided_tensor_core.py in [`StridedTensorCore.__init__`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor_core.py#L27)
- colbert/modeling/checkpoint.py in [`Checkpoint.score](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L215)
- colbert/indexing/utils.py in [`optimize_ivf`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/utils.py#L48)

ColBERT uses `torch.cumsum` in the following files to calculate `offsets`.:

- [colbert/indexing/utils.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/utils.py#L50)
- [colbert/search/strided_tensor_core.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor_core.py#L31)
- [colbert/search/strided_tensor.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L48)
- [colbert/search/index_storage.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L68)

ColBERT uses `torch.where` in the following files:

- [colbert/modeling/checkpoint.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L49) to pool embeddings within each cluster.


ColBERT uses `torch.nonzero` in the following files:

- [colbert/index_updater.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L351) to construct mask of where pids to be removed appear in ivf.

<mark>These are all MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PRs [#91120](https://github.com/pytorch/pytorch/pull/91120), [#94464](https://github.com/pytorch/pytorch/pull/94464)

> Fix issues with torch.bool for Unary ops (#91120), scatter ops (#94464)

Claude's analysis: Claude: The PR fixes compatibility issues where boolean tensors needed to be cast to int8 on older macOS versions, then cast back. This would be important for ColBERT's masking operations which rely heavily on boolean tensors for attention and padding masks.

<mark>These are MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PR [#94484](https://github.com/pytorch/pytorch/pull/94484)

> Properly cast torch.int64 to torch.int32 for reduction ops and raise warning. (#94484)

Claude's analysiss: The PR changes TORCH_CHECK (which throws an error) to TORCH_WARN_ONCE (which just warns) and automatically casts int64 to int32 for min/max operations. This would allow ColBERT to run on MPS with int64 tensors instead of failing, though with potential precision loss.

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PRs [#91120](https://github.com/pytorch/pytorch/pull/91197), [#94464](https://github.com/pytorch/pytorch/pull/91514)

> Fix handling of ops taking multiple dtypes as input (#91197, #91514)

Claude's analysis: The PR fixes MPS scatter to handle type mismatches between source and destination tensors automatically.

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PRs [#91786](https://github.com/pytorch/pytorch/pull/91786), [#94662](https://github.com/pytorch/pytorch/pull/94662)

> Fix handling of channels last for torch.cat (#91786, #94662), torch.Conv2d (#91822, #94384), torch.nn.{ELU,ReLU,Hardswish} (#94664), torch.nn.BatchNorm (#94760), torch.nn.MaxPool2d (#94877)

ColBERT uses `.cat` in the following files:

- [colbert/indexing/utils.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/utils.py#L45) to concatenate a list of tensors (`unique_pids_per_centroid`) into a single tensor (`ivf`).
- [colbert/indexing/index_manager.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/index_manager.py#L23) to concatenate multiple path names.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor_core.py#L31) to calculate `offsets`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor_core.py#L39) to add padding to a tensor.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/distillation/scorer.py#L60) to concatenate `scores`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/collection_encoder.py#L38) to concatenate document token embeddings.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L181) to concatenate `codes` (centroid IDs).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L182) to concatenate `residuals`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L220) to concatenate `codes` (centroid IDs).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L237) to concatenate `centroids`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L276) to concatenate document token embeddings.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L51) to concatenate `packed_tensor`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L152) to concatenate `all_orders`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/strided_tensor.py#L155) to concatenate `all_lengths`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L101) to concatenate `compressed_embs.codes` (centroid IDs corresponding to document token embeddings).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L108) to concatenate `compressed_embs.residuals`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L117) to concatenate `doclens`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L431) to concatenate `codes` (centroid IDs).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L434) to concatenate `residuals`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L506) to concatenate `codes` (centroid IDs).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/index_updater.py#L507) to concatenate `residuals`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L115) to concatenate `batches`  (of queries).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L168) to concatenate document token embeddings (in order).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L169) to concatenate `mask` for document token embeddings (in order).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/coalesce.py#L48) to concatenate `code` chunks.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L69) to concatenate `offsets`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L149) to concatenate `approx_scores`.
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/tokenization/query_tokenization.py#L88) to concatenate `ids` (for query tokens).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/tokenization/query_tokenization.py#L89) to concatenate `masks` (for query tokens).
- [](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/tokenization/utils.py#L72) to concatenate prefix token.

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PRs [#94259](https://github.com/pytorch/pytorch/pull/94259), [#94278](https://github.com/pytorch/pytorch/pull/94278), [#95145](https://github.com/pytorch/pytorch/pull/95145), [#95762](https://github.com/pytorch/pytorch/pull/95762), [#95905](https://github.com/pytorch/pytorch/pull/95905)

> Fix view operations handling (#94259, #94278,#95145, #95762, #95905)

Claude's analysis: This PR fixes crashes in view operations when slicing with incorrect lengths, which ColBERT uses for tensor reshaping and indexing operations.

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PR [#87853](87853)

> Move incorrectly placed closing curly brace of extern "C" block (#87853)

Gemini's analysis: This pull request is a foundational C++ correctness fix for the PyTorch framework. Because ColBERT compiles its own C++ extensions that depend on these core headers, this change is directly beneficial. It ensures the stability and reliability of ColBERT's own build process, preventing potential compilation failures.

ColBERT's C++ extensions:

- [segmented_maxsim.cpp](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/segmented_maxsim.cpp)
- [filter_pids.cpp](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/search/filter_pids.cpp)
- [decompress_residuals.cpp](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/search/decompress_residuals.cpp)

<mark>Unsure how to measure the impact but if it's about reliability perhaps it will address some open issues. TBD.</mark>

### PR [#93322](https://github.com/pytorch/pytorch/pull/93322)

> Fix MSVC compiler error in basic_ops.h (#93322)

Gemini's take: This pull request is a crucial build-system and correctness fix. It directly impacts ColBERT by ensuring that its custom C++ code can be compiled successfully on Windows machines that use the affected MSVC compiler. Without this fix, users in that environment would be unable to run ColBERT. This change makes ColBERT's build process more robust and widens its platform compatibility.

<mark>TBD if this addressed open issues related to Windows machines.</mark>

### PR [#89310](https://github.com/pytorch/pytorch/pull/89310)

> Fix a bug that redefines __STDC_FORMAT_MACROS (#89310)

Gemini's take: This pull request provides a stability and correctness fix to the underlying PyTorch framework. Because ColBERT compiles its own C++ code that depends on these core PyTorch headers, this change is directly beneficial. It makes ColBERT's own compilation process more reliable and prevents a potential class of build failures.

<mark>Unsure how to measure the impact but if it's about reliability perhaps it will address some open issues. TBD.</mark>

### PR [#90411](https://github.com/pytorch/pytorch/pull/90411)

> Add manual cuda deps search logic (#90411)

Gemini's take: This PyTorch pull request adds a new mechanism to help PyTorch find its essential CUDA libraries (cuBLAS and cuDNN) on Linux systems.

<mark>Unsure how to measure the impact but if it's about reliability perhaps it will address some open issues. TBD.</mark>

### PR [#89759](https://github.com/pytorch/pytorch/pull/89759)

> Workaround for NumPy builds that ship with a broken Dlpack deleter (#89759)

<mark>TBD if this improves reliability as ColBERT uses NumPy</mark>/.

### PR [#86288](https://github.com/pytorch/pytorch/pull/86288)

> Workaround MSVC ICE due to constexpr char* template argument (#86288)

Gemini's take: It directly impacts ColBERT by ensuring that its custom C++ code can be compiled successfully on Windows machines that use an affected MSVC compiler.

<mark>TBD if this addressed open issues related to Windows machines.</mark>

### PR [#85408](https://github.com/pytorch/pytorch/pull/85408)

> Add define to fix issue with compatibility with latest Windows SDK (#85408)

Gemini's take: It directly impacts ColBERT by ensuring that the underlying PyTorch framework can be successfully built on modern Windows environments. 

<mark>TBD if this addressed open issues related to Windows machines.</mark>

## Improvements

These PyTorch PRs are related to improvements, which could affect ColBERT by speeding things up (and therefore seeing a speed up in indexing/search/training time) or changing baseline indexing/search/training artifacts if improvements impact numeric precision.

### PR [#56398](https://github.com/pytorch/pytorch/pull/56398)

> Set std/var correction overloads default value to None (#56398)

<mark>Unclear if and how this affects ColBERT but highlighting it since it changes code in PyTorch's aten/src/ATen/native.</mark>

### PR [#86309](https://github.com/pytorch/pytorch/pull/86309)

> Add support for int32 indices in index/index_put ops (#86309) 

<mark>I think this PR is related to [this ColBERT PR](https://github.com/stanford-futuredata/ColBERT/pull/180) which I think is related to [this line of code in `IndexScorer`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L163)</mark>.

### PR [#87022](https://github.com/pytorch/pytorch/pull/87022)

> Enable where to have cpu scalar args (#87022)

ColBERT uses `torch.where` in the following files:

- [colbert/modeling/checkpoint.py](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/checkpoint.py#L49) to pool embeddings within each cluster.

<mark>Unclear if this will affect ColBERT but there are currently no open issues related to `torch.where`</mark>.

### PR [#90914](https://github.com/pytorch/pytorch/pull/90914)

> Add support for NumPy scalars to torch.tensor.asarray (#90914)

<mark>Found [1 use of `asarray`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L198) but it doesn't deal with a scalar so probably won't be affected</mark>.

### PR [#85926](https://github.com/pytorch/pytorch/pull/85926)

> Enable out variant of torch.max(#85926)

<mark>Unclear what this PR does but highlighting it since ColBERT uses `torch.max`</mark>.

### PR [#91846](https://github.com/pytorch/pytorch/pull/91846)

> Implement faster gradient clipping using foreach function (#91846)

ColBERT uses `torch.nn.utils.clip_grad_norm_` in two lines:

- [colbert/utils/amp.py#L26](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/amp.py#L26)
- [colbert/utils/amp.py#L31](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/amp.py#L31)
 
<mark>IIUC this won't affect ColBERT since it doesn't set `foreach` in `torch.nn.utils.clip_grad_norm_`.</mark>

### PR [#92334](https://github.com/pytorch/pytorch/pull/92334)

> Enable DDP to handle custom dataclass forward outputs (#92334)

<mark>ColBERT does use DistributedDataParallel ([in `train`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/training/training.py#L56)) butit's not being passed a custom dataclass, it's being passed a `colbert` model so I don't think this PR applies.</mark>

### PR [#89137](https://github.com/pytorch/pytorch/pull/89137)

> Skip collective communications for NO_SHARD in clip_grad_norm_ (#89137)

<mark>ColBERT doesn't use FullyShardedDataParallel but it [does use `torch.nn.utils.clip_grad_norm`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/amp.py#L26) so not sure if this PyTorch PR affects it</mark>.

### PR [#90028](https://github.com/pytorch/pytorch/pull/90028)

> Apply the "largest" dtype across all parameters/gradients as defined by PyTorch's type promotion semantics for the total norm returned in clip_grad_norm_ for low prec grads (#90028)

<mark>ColBERT doesn't use FullyShardedDataParallel but it [does use `torch.nn.utils.clip_grad_norm`](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/utils/amp.py#L26) so not sure if this PyTorch PR affects it</mark>.

### PR [#85692](https://github.com/pytorch/pytorch/pull/85692)

> Set CUDA_MODULE_LOADING to LAZY when not set by the user (#85692)

<mark>Unclear exactly what this does but it relates to the CUDA_MODULE_LOADING env var which is not set in ColBERT</mark>

### PR [#89172](https://github.com/pytorch/pytorch/pull/89172)

> Add an option to disable reduced precision reductions for BF16 GEMM (#89172)

<mark>Unclear exactly what this does, but in the PR they mentioned it improves H100 usage, so I'll keep that in mind.</mark>

### PR [#91436](https://github.com/pytorch/pytorch/pull/91436)

> Add an env variable to disable addmm_cuda_lt kernel (#91436)

<mark>Unclear what this does, but it's adding a variable, so it's a new feature.</mark>

### PR [#86041](https://github.com/pytorch/pytorch/pull/86041),  [#93022](https://github.com/pytorch/pytorch/pull/93022)

> Clean up flatbuffer lib dependency and fixed its test to match pkl models (#86041, #93022)

<mark>I am not sure what these PRs are doing. The title refers "pkl models" which ColBERT doesn't use to my knowledge.</mark>

### PR [#93898](https://github.com/pytorch/pytorch/pull/93898)

> Type corrections to avoid unnecessary static_casts (#93898)

<mark>Unclear what this PR does but it touches a lot of what seem to be core files so I'm flagging it</mark>.

### PR [#87245](https://github.com/pytorch/pytorch/pull/87245)

> Integrate all ONNX operators with a new JitScalarType API (#87245)

<mark>It's onnx related, which ColBERT doesn't use, but it also says: "this PR addresses not only the issue above, but the entire family of issues related to torch._C.Value.type() parsing when scalarType() or dtype() is not available."</mark>

### PR [#87343](https://github.com/pytorch/pytorch/pull/87343)

> Add share_from_this to torch::jit::Graph (#87343)

<mark>Is ONNX related, but unclear if it affects anything else?</mark>

### PR [#84789](https://github.com/pytorch/pytorch/pull/84789)

> Use optional op to keep None in results for ONNX internal tests (#84789)

<mark>Is ONNX related, but unclear if it affects anything else?</mark>

### PR [#86218](https://github.com/pytorch/pytorch/pull/86218)

> Add fp16 support for torch.nn.Linear (#89774), torch.nn.GELU (#86218)

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PR [#91884](https://github.com/pytorch/pytorch/pull/91884)

> Add support for empty Tensors in torch.bitwise_not (#87286), torch.nn.LayerNorm (#94212), many backward functions (#94343), torch.nn.functional.hardswish (#94342), torch.topk (#91884), torch.arange (#94485), torch.linal.inv (#94551),

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>


### PR [#91734](https://github.com/pytorch/pytorch/pull/91734)

> Add support for reduction ops on multiple axis at a time (#91734)

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PR [#94639](https://github.com/pytorch/pytorch/pull/94639)

> Add support for k greater than 16 for torch.topk (#94639)

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

### PR [#91576](https://github.com/pytorch/pytorch/pull/91576)

> Simplify OpenMP detection in CMake (#91576)

Claude's take: While ColBERT should continue working, there could be subtle performance or compilation differences depending on how PyTorch's simplified OpenMP detection affects the runtime compilation of ColBERT's C++ extensions, particularly in multi-threaded scenarios.

<mark>Unclear what this PR is doing but flagging it as it might improve performance as Claude states.</mark>

## Deprecations

These are PRs I would think would have a significant impact if applicable. 

### PR [#92143](https://github.com/pytorch/pytorch/pull/92143)

> Deprecate tensor.mT,tensor.T,tensor.mH,tensor.H on 0D-tensors (#92143)

<mark>There are five instances where .T is used, but pretty sure none of these are 0-D tensors, will confirm</mark>:

- [colbert/search/candidate_generation.py#L13](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L13): cosine similarity between centroids and query token embeddings.
- [colbert/search/candidate_generation.py#L43](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/candidate_generation.py#L43): used in `generate_candidate_scores` which uses `lookup_eids` which I can't find anywhere else in the codebase.
- [colbert/modeling/colbert.py#L195](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/modeling/colbert.py#L195): Cosine similarity between query and document token embeddings. 
- [colbert/indexing/codecs/residual.py#L215](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L215): Cosine similarity between centroids and document token embeddings (GPU).
- [colbert/indexing/codecs/residual.py#L217](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/codecs/residual.py#L217): Cosine similarity between centroids and document token embeddings (CPU).

## Performance

Similar to improvements, I would only expect this set of PRs to improve ColBERT performance, keeping an eye on how different artifacts changed because of that. 

### PR [#93234](https://github.com/pytorch/pytorch/pull/93234)

> Improve performance for functional.multi_head_attention_forward() (#93234, #89847)

<mark>ColBERT uses BERT, which has its own attention implementation, so this likely wouldn't impact it unless the BERT model specifically uses `torch.nn.functional.multi_head_attention_forward` or `torch.nn.MultiheadAttention`.</mark>

### PR [#84981](https://github.com/pytorch/pytorch/pull/84981)

> Use atomicAdd for bfloat16 in Ampere and above (#84981)

Gemini's take: This pull request directly accelerates a fundamental operation used during the training of ColBERT. By replacing a slow, emulated function with a fast, hardware-native instruction, this change leads to a noticeable increase in training speed for anyone training ColBERT with bfloat16 mixed precision on an Ampere or newer GPU. 

<mark>If Gemini is correct, then I will see a speedup in training.</mark>

### PR [#94034](https://github.com/pytorch/pytorch/pull/94034)

> Add various performance fixes to c++ STL usage (#94034)

Gemini's take: The changes in this PR touch several core PyTorch components that are critical to ColBERT's operation:

- Autograd Engine (function.h): Every gradient calculation during training will benefit from these optimizations.
- CUDA Communication (comm.cpp): The code that handles broadcasting and gathering tensors across GPUs for multi-GPU training and inference is made more efficient.
- Mixed Precision (autocast_mode.h): The logic for automatic mixed precision, which is key for training ColBERT efficiently, is also slightly optimized.

<mark>If Gemini is correct, then I will see a speed up in all aspects of ColBERT. </mark>

### PR [#86568](https://github.com/pytorch/pytorch/pull/86568)

> Add fmsub to vectorization primitives (#86568)

Gemini's take: This pull request is a CPU-specific performance optimization. It adds support for the fmsub (fused multiply-subtract) instruction to PyTorch's CPU vectorization library. This allows PyTorch to perform the operation (a * b) - c in a single, faster instruction on modern CPUs that support it (e.g., via AVX or NEON).

<mark>I'm pretty sure ColBERT doesn't use multiply-subtract, but keeping it in here just in case it comes up. </mark>

### PR [#92300](https://github.com/pytorch/pytorch/pull/92300)

> Fix biasadd OMP perf issue for the packed MKL SGEMM (#92300)

Gemini's take: This pull request is a CPU-specific performance optimization. It fixes a parallelization issue within the Intel MKL (Math Kernel Library) backend for linear layers. This change improves the efficiency of adding a bias term to the output of a matrix multiplication when running on a CPU.

<mark>If Gemini is correct, I would expect a speedup on CPU. </mark>


### PR [#91114](https://github.com/pytorch/pytorch/pull/91114)

> Increase performance of torch.add{cmul,cdiv,mm}(#94214, #94534)torch.multinomial (#86342), faster op launch time (#86437), torch.linear (#91114), view handling (#91743, #94218), convolutions(#94661), scatter/gather (#94663)

Gemini's take: While the Adam optimizer used by ColBERT does use the addcdiv operation, this is executed on the GPU via CUDA, not MPS. This pull request is a performance optimization for the torch.nn.Linear layer, but it is exclusively for the MPS (Metal Performance Shaders) backend. 

<mark>MPS-related and will only affect ColBERT if we in the future choose to make it compatible with MPS.</mark>

## Closing Thoughts

Based on my analysis, I'm optimistic about upgrading ColBERT from torch==1.13.1 to 2.0. The upgrade should deliver concrete benefits with reasonable testing overhead. Performance-wise, I'll be watching for training time improvements from the faster `foreach` optimizer implementations and expect speedups across all aspects of ColBERT from C++ optimizations and CUDA improvements. For validation, I'll need to check for numeric differences in indexing/search artifacts from half-precision bug fixes and benchmark retrieval quality metrics after reindexing to avoid regressions. The reliability improvements should make ColBERT's multi-GPU functionality more reliable and might address related open issues. Plus there are fixes for operations like `topk` and `torch.load` that ColBERT uses extensively. Most MPS-related changes will only affect ColBERT if we choose future compatibility, so they're not immediate concerns but good to have.

My next step will be to establish training time benchmarks and indexing/retrieval/training baseline artifacts so that I can concretely monitor even subtle performance/behavior changes when using `torch==2.0` in my development branch.

---
title: PyTorch Version Impact on ColBERT Index Artifacts
date: "2025-08-19"
author: Vishal Bakshi
description: Analysis of how ColBERT index artifacts change when upgrading PyTorch from 1.13.1 to 2.1.0. Differences in index tensors root cause is likely floating point variations in BERT model forward passes.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

I recently released `colbert-ai==0.2.22` which removed the deprecated `transformers.AdamW` import [among other changes](https://github.com/stanford-futuredata/ColBERT/releases/tag/v0.2.22). I'm now turning my attention to upgrading the PyTorch dependency to 2.x, which will not only introduce compatibility with modern version installations of `torch` but will also allow the integration of the [AnswerAI `fastkmeans` library](https://github.com/AnswerDotAI/fastkmeans) as a replacement for the `faiss-gpu` and `faiss-cpu` libraries (which are no longer officially maintained on PyPI).

I started this PyTorch 2.x upgrade effort by analyzing the impact of `torch==2.0.0` on `colbert-ai` as this was the first upgrade from the existing `torch==1.13.1` dependency. I approached this analysis by reviewing and documenting whether the 500+ PRs involved in `torch==2.0.0` would impact `colbert-ai`. The resulting [spreadsheet](https://docs.google.com/spreadsheets/d/1sUEN7xo5-hLVoxF9NL_ibGxPaKlPzxmnMU46zf3wd-U/edit?usp=sharing) and [blog post](https://vishalbakshi.github.io/blog/posts/2025-07-27-torch-colbert/) detail my findings. In short, I estimated that 28 PRs potentially impacted `colbert-ai`.

In this blog post I'm detailing a different approach, from the "other end" so to speak: what changes in `colbert-ai` index artifacts when changing PyTorch versions?

## Indexing ConditionalQA with 19 PyTorch Versions

I started by indexing the [UKPLab/DAPR/ConditionalQA](https://huggingface.co/datasets/UKPLab/dapr) document collection with 19 different `colbert-ai` installs (one for each version of PyTorch from `1.13.1` to `2.8.0`), using Modal. Each Dockerfile looks something like this:

```Dockerfile
FROM mambaorg/micromamba:latest

USER root

RUN apt-get update && apt-get install -y git nano curl wget build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/stanford-futuredata/ColBERT.git /ColBERT && \
    cd /ColBERT && \
    micromamba create -n colbert python=3.11 cuda -c nvidia/label/11.7.1 -c conda-forge && \
    micromamba install -n colbert faiss-gpu -c pytorch -c conda-forge && \
    micromamba run -n colbert pip install -e . && \
    micromamba run -n colbert pip install torch==2.2.0 transformers==4.38.2 pandas

ENV CONDA_DEFAULT_ENV=colbert
ENV PATH=/opt/conda/envs/colbert/bin:$PATH

WORKDIR /ColBERT

RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
    echo "micromamba activate colbert" >> ~/.bashrc

CMD ["/bin/bash"]
```

I decided to `git clone` and `pip install -e .` the `main` branch of [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/colbert) since I wanted to modify the files down the road to save/inject intermediate index artifacts (as we'll see later on in this blog post).

My indexing function looks like:

```python
@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _index(source, project, date, nranks, ndocs, root):
    import os
    import subprocess
    subprocess.run(['pwd'], text=True, shell=True)
    from colbert import Indexer
    from colbert.infra import RunConfig, ColBERTConfig
    from colbert.infra.run import Run
    from datasets import load_dataset

    os.environ["ROOT"] = root

    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")
    qrels_rows = load_dataset("UKPLab/dapr", f"{dataset_name}-qrels", split="test")

    with Run().context(RunConfig(nranks=nranks)):
        config = ColBERTConfig(
            doc_maxlen=256,      
            nbits=4,             
            dim=96,             
            kmeans_niters=20,
            index_bsize=32,
            bsize=64,
            checkpoint="answerdotai/answerai-colbert-small-v1"
        )
        
        indexer = Indexer(checkpoint="answerdotai/answerai-colbert-small-v1", config=config)
        _ = indexer.index(name=f"{MOUNT}/{project}/{date}-{source}-{nranks}/indexing/{dataset_name}", collection=passages[:ndocs]["text"], overwrite=True)

    print("Index created!")
```

I would run the indexing function (in my `main.py` file) using a terminal command like so:

```bash
SOURCE="0.2.22.main.torch.1.13.1" DATE="20250818" PROJECT="torch2.x" NRANKS=1 GPU="L4" modal run main.py
```

## Comparing Index Artifacts Across PyTorch Versions

Once indexed, I ran my comparison script which starts by comparing index file names:

```python
console.print("\n[bold blue]INDEX FILE NAME COMPARISON[/bold blue]")
a = os.listdir(a_path)
b = os.listdir(b_path)

try:
    for i, f in enumerate(a): assert f == b[i]
    console.print(f"[green]✓ All {len(a)} files match[/green]")
except:
    console.print("[red]✗ File names don't match[/red]")
```

Then index tensor shapes:

```python
for i, f in enumerate(a_pts):
    console.print(f"\n[bold]{f}[/bold]")
    a_pt = torch.load(a_path + f)
    b_pt = torch.load(b_path + f)
    
    if isinstance(a_pt, tuple):
        match1 = a_pt[0].shape == b_pt[0].shape
        match2 = a_pt[1].shape == b_pt[1].shape
        console.print(f"  Tensor[0]: [{'green' if match1 else 'red'}]{a_pt[0].shape} vs {b_pt[0].shape}[/{'green' if match1 else 'red'}]")
        console.print(f"  Tensor[1]: [{'green' if match2 else 'red'}]{a_pt[1].shape} vs {b_pt[1].shape}[/{'green' if match2 else 'red'}]")
        if not (match1 and match2):
            shape_mismatches += 1
    else:
        match = a_pt.shape == b_pt.shape
        console.print(f"  Shape: [{'green' if match else 'red'}]{a_pt.shape} vs {b_pt.shape}[/{'green' if match else 'red'}]")
        if not match:
            shape_mismatches += 1
```

and finally compare tensor values between indexes:

```python
for i, f in enumerate(a_pts):
    console.print(f"\n[bold]{f}[/bold]")
    a_pt = torch.load(a_path + f)
    b_pt = torch.load(b_path + f)
    
    if isinstance(a_pt, tuple):
        if a_pt[0].shape == b_pt[0].shape:
            match1 = torch.allclose(a_pt[0], b_pt[0])
            console.print(f"  [{'green' if match1 else 'red'}]{'✓' if match1 else '✗'} Tensor[0] values {'match' if match1 else 'differ'}[/{'green' if match1 else 'red'}]")
        else:
            console.print("  [red]✗ Tensor[0] shape mismatch[/red]")
            match1 = False
            
        if a_pt[1].shape == b_pt[1].shape:
            match2 = torch.allclose(a_pt[1], b_pt[1])
            console.print(f"  [{'green' if match2 else 'red'}]{'✓' if match2 else '✗'} Tensor[1] values {'match' if match2 else 'differ'}[/{'green' if match2 else 'red'}]")
        else:
            console.print("  [red]✗ Tensor[1] shape mismatch[/red]")
            match2 = False
            
        if not (match1 and match2):
            value_mismatches += 1
    else:
        if a_pt.shape == b_pt.shape:
            match = torch.allclose(a_pt, b_pt)
            console.print(f"  [{'green' if match else 'red'}]{'✓' if match else '✗'} Values {'match' if match else 'differ'}[/{'green' if match else 'red'}]")
        else:
            console.print("  [red]✗ Shape mismatch[/red]")
            match = False
            
        if not match:
            value_mismatches += 1
```

I compared consecutive pairs of PyTorch version `colbert-ai` installs to understand between which versions the index artifacts change. Here are my results:


|Version A|Version B|All `.pt` Shapes Match? (Matches)|All `.pt` Values Match? (Matches)|
|:-:|:-:|:-:|:-:|
|1.13.1|2.0.0|Yes (10/10)|Yes (10/10)|
|2.0.0|2.0.1|Yes (10/10)|Yes (10/10)|
|<mark>2.0.1</mark>|<mark>2.1.0</mark>|<mark>No (9/10)</mark>|<mark>No (0/10)</mark>|
|2.1.0|2.1.1|Yes (10/10)|Yes (10/10)|
|2.1.1|2.1.2|Yes (10/10)|Yes (10/10)|
|2.1.2|2.2.0|Yes (10/10)|Yes (10/10)|
|2.2.0|2.2.1|Yes (10/10)|Yes (10/10)|
|2.2.1|2.2.2|Yes (10/10)|Yes (10/10)|
|2.2.2|2.3.0|Yes (10/10)|Yes (10/10)|
|2.3.0|2.3.1|Yes (10/10)|Yes (10/10)|
|2.3.1|2.4.0|Yes (10/10)|Yes (10/10)|
|2.4.0|2.4.1|Yes (10/10)|Yes (10/10)|
|<mark>2.4.1</mark>|<mark>2.5.0</mark>|<mark>No (9/10)</mark>|<mark>No (0/10)</mark>|
|2.5.0|2.5.1|Yes (10/10)|Yes (10/10)|
|2.5.1|2.6.0|Yes (10/10)|Yes (10/10)|
|2.6.0|2.7.0|Yes (10/10)|Yes (10/10)|
|2.7.0|2.7.1|Yes (10/10)|Yes (10/10)|
|<mark>2.7.1</mark>|<mark>2.8.0</mark>|<mark>Yes (10/10)</mark>|<mark>No (6/10)</mark>|

There are three PyTorch upgrades that cause a change in index artifacts: 2.0.1 --> 2.1.0, 2.4.1 --> 2.5.0, and 2.7.1 --> 2.8.0.

## Comparing Intermediate Index Artifacts

To better understand exactly where the index artifacts changed when upgrading PyTorch, I created my own copies of two stanford-futuredata/ColBERT files and added `torch.save` lines to save the intermediate artifacts listed below:

- colbert/indexing/collection_indexer.py
    - `sampled_pids` (a set of integers corresponding to sampled passage IDs)
    - `num_passages` (a single integers, the number of total passages in the collection)
    - `local_sample_embs` (BERT encodings of the sample pids, created by `Checkpoint.docFromText`)
    - `centroids` (from `_train_kmeans`)
    - `bucket_cutoffs` (the bin "boundaries" used for quantization from ` _compute_avg_residual`)
    - `bucket_weights` (the quantized values, from ` _compute_avg_residual`)
    - `avg_residual` (a single float, from ` _compute_avg_residual`)
    - `sample` (95% of the values from `local_sample_embs.half()`)
    - `sample_heldout` (5% of the values from `local_sample_embs.half()`)
    - `embs` (encoded passages)
    - `doclens` (number of tokens in each passage)
    - `codes` (centroid IDs (values) and document token IDs (indices))
    - `ivf` (document token IDs)
    - `values` (centroid IDs)
- colbert/modeling/checkpoint.py
    - `tensorize_output` (tuple (`text_batches`, `reverse_indices`) output from `DocTokenizer.tensorize`)
    - `batches` (BERT encodings, output from `Checkpoint.doc`)
    - `D` (sorted and reshaped `batches`)

I then replaced the corresponding files in the `/ColBERT` directory (which is why I used `git clone` and `pip install e .`) with the following lines for Modal:

```python
image = image.add_local_file("collection_indexer.py", "/ColBERT/colbert/indexing/collection_indexer.py")
image = image.add_local_file("checkpoint.py", "/ColBERT/colbert/modeling/checkpoint.py")
```

Here are the results when comparing these artifacts between `colbert-ai` installs using `torch==1.13.1` and `torch==2.1.0`:

|Artifact|`torch.allclose`|
|:-:|:-:|:-:|
|`sampled_pids`|`True`
|`num_passages`|`True`
|<mark>`local_sample_embs`</mark>|<mark>`False`</mark>
|<mark>`centroids`</mark>|<mark>`False`</mark>
|<mark>`bucket_cutoffs`</mark>|<mark>`False`</mark>
|<mark>`bucket_weights`</mark>|<mark>`False`</mark>
|<mark>`avg_residual`</mark>|<mark>`False`</mark>
|<mark>`sample`</mark>|<mark>`False`</mark>
|<mark>`sample_heldout`</mark>|<mark>`False`</mark>
|<mark>`embs`</mark>|<mark>`False`</mark>
|`doclens`|`True`
|<mark>`codes`</mark>|<mark>`False`</mark>
|<mark>`ivf`</mark>|<mark>`False`</mark>
|<mark>`values`</mark>|<mark>`False`</mark>
|`tensorize_output`|`True`
|<mark>`batches`</mark>|<mark>`False`</mark>
|<mark>`D`</mark>|<mark>`False`</mark>

After reviewing these comparisons, my hypothesis was that the first difference (in `local_sample_embs`) affected all subsequent artifacts. The difference in `local_sample_embs` can be traced down to the difference in `batches` and `D`. To test this hypothesis, I "injected" the `local_sample_embs` from the `torch==1.13.1` install into the `collection_indexer.py` when indexing with `torch==2.1.0`:

```python
local_sample_embs = torch.load("/colbert-maintenance/torch2.x/20250818-0.2.22.main.torch.1.13.1-1k-1/local_sample_embs.pt")
```

I then re-compared the artifacts, and my hypothesis was correct!

|Artifact|`torch.allclose`|
|:-:|:-:|:-:|
|`centroids`|`True`|
|`bucket_cutoffs`|`True`|
|`bucket_weights`|`True`|
|`avg_residual`|`True`|
|`sample`|`True`|
|`sample_heldout`|`True`|
|<mark>`embs`</mark>|<mark>`False`</mark>
|`doclens`|`True`
|<mark>`codes`</mark>|<mark>`False`</mark>
|<mark>`ivf`</mark>|<mark>`False`</mark>
|<mark>`values`</mark>|<mark>`False`</mark>

## Comparing the `BertModel`s

Where do `local_sample_embs` come from? The highest-level method is `CollectionEncoder.encode_passages`. Inside `CollectionEncoder.encode_passages` the collection of texts, `passages` is fed to `Checkpoint.docFromText`. Inside there, the tokenized text is passed to `Checkpoint.doc`, which passes them to `ColBERT.doc`, which finally passes the `input_ids` and `attention_mask` to `ColBERT.bert`. Since there was a divergence in `local_sample_embs`, I figured there would be a divergence in either the weights and/or the logits of `ColBERT.bert` between both PyTorch version installs.

I installed each image of `colbert-ai` and separately saved the `BertModel` weights as well as a dictionary with outputs from each of the 10 `BertEncoder` layers. These outputs were accessed using a forward hook:

```python
outputs_dict = {}

def capture_output(name):
    def hook_fn(module, input, output):
        outputs_dict[name] = output[0].detach()
    return hook_fn

hooks = []
for i in range(10):
    hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"1.13.1_{i}")))

with torch.no_grad():
    D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]

for h in hooks: h.remove()
```

Both `colbert-ai` installs (`torch==1.13.1` and `torch==2.1.0`) had equal `BertModel` weights. However, both of them have diverging `BertEncoder` outputs.

Here are the mean absolute differences between corresponding `BertEncoder` layer outputs between `torch==1.13.1` and `torch==2.1.0`:

```python
for i in range(10):
    a_ = a[f"1.13.1_{i}"]
    b_ = b[f"2.1_{i}"]
    print(i, torch.abs(a_ - b_).float().mean())
```

```
0 tensor(2.8141e-08, device='cuda:0')
1 tensor(5.9652e-08, device='cuda:0')
2 tensor(8.0172e-08, device='cuda:0')
3 tensor(7.8228e-08, device='cuda:0')
4 tensor(7.9968e-08, device='cuda:0')
5 tensor(8.3589e-08, device='cuda:0')
6 tensor(8.7348e-08, device='cuda:0')
7 tensor(8.5140e-08, device='cuda:0')
8 tensor(8.5651e-08, device='cuda:0')
9 tensor(8.1636e-08, device='cuda:0')
```

The difference increases about 2x as we go deeper through the model.

Here are the max absolute differences, which increases 2x by the final layer:

```python
0 tensor(3.5763e-07, device='cuda:0')
1 tensor(4.7684e-07, device='cuda:0')
2 tensor(5.9605e-07, device='cuda:0')
3 tensor(5.9605e-07, device='cuda:0')
4 tensor(7.1526e-07, device='cuda:0')
5 tensor(7.1526e-07, device='cuda:0')
6 tensor(7.1526e-07, device='cuda:0')
7 tensor(9.5367e-07, device='cuda:0')
8 tensor(9.5367e-07, device='cuda:0')
9 tensor(1.1921e-06, device='cuda:0')
```

## Closing Thoughts

From this analysis, I can conclude that the difference in index artifacts generated by `colbert-ai` using different `torch==1.13.1` vs. `torch==2.1.0` is due to floating point differences in the forward pass of the `BertModel` used to generate token-level embeddings from text passages. I have not yet analyzed the `torch==2.1.0` release notes to make an educated guess on why these differences occur. But given that it's during the forward pass of the model, I would wager there was some update to the underlying C++ code for the `torch.nn` module.

I will move forward with comparing intermediate artifacts between each subsequent versions where the final index artifacts are different 2.4.1 --> 2.5.0, and 2.7.1 --> 2.8.0. Once that's complete, I'll dive into the PyTorch release notes and see if I can reasonably point to a few PRs behind this change. Once I have a reasonable handle on understanding `colbert-ai` indexing behavior with different versions of PyTorch 2.x, I'll perform a similar analysis with `colbert-ai` training and document my findings. 

Thanks for reading until the end! I'll be posting more blog post and/or video updates around ColBERT maintenance as soon as I have something more to share.
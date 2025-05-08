---
title: TIL&#58; Resolving RAGatouille OOM Error and `faiss-gpu` Warning
date: "2025-05-08"
author: Vishal Bakshi
description: A couple of fixes as I work on indexing large document collections (6M+) using RAGatouille.
filters:
   - lightbox
lightbox: auto
categories:
    - information retrieval
    - deep learning
---

I'm in the process of indexing the UKPLab/DAPR datasets, which span in size from ~70k to ~32M documents. Using a RTX3090, I ran into an OOM error (during search) and a warning stating that faiss-cpu was being used instead of faiss-gpu, causing the indexing process to take longer.

I found [this RAGatouille GitHub issue](https://github.com/AnswerDotAI/RAGatouille/issues/177) which recommended lowering the `batch_size` in ColBERT's [`IndexScorer.score_pids` method](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L121). I made that change (from 2^20 to 2^16) and that resolved the OOM error, at least for the 2.68M document collection (NaturalQuestions).

When I was using Google Colab GPUs, the following install commands correctly installed faiss-gpu after installing RAGatouille:

```python
pip uninstall -y faiss-cpu
pip install faiss-gpu-cu12
```

Using an RTX3090 (not on Colab), this was not correctly installing faiss-gpu, leading to the following RAGatouille warning during indexing, and as a result, using the CPU for indexing (which eventually crashed the kernel):

```
________________________________________________________________________________
WARNING! You have a GPU available, but only `faiss-cpu` is currently installed.
This means that indexing will be slow. To make use of your GPU
Please install `faiss-gpu` by running:
pip uninstall --y faiss-cpu & pip install faiss-gpu
________________________________________________________________________________
```

This warning is thrown in RAGatouille's [`PLAIDModelIndex.build`](https://github.com/AnswerDotAI/RAGatouille/blob/2bd4d2ed01c847854be78704a012f9ab35d679b2/ragatouille/models/index.py#L226) if `hasattr(faiss, "StandardGpuResources")` is `False`.

Looking at the [faiss repo](https://github.com/facebookresearch/faiss/tree/main#:~:text=faiss%2Dcpu%2C-,faiss%2Dgpu,-and%20faiss%2Dgpu), they recommend using conda for installation. I ran `conda install pytorch::faiss-gpu`, restarted the kernel, confirmed that `hasattr(faiss, "StandardGpuResources")` returns `True` and was successfully able to circumvent that warning. As a result, RAGatouille was able to use faiss-gpu and it was able to index 2M document.

It's still TBD if this allows me to finish indexing all of my datasets (especially the 13M and 32M ones).

In a conversation with Claude, I outlined a few different scenarios that I may have to (get to) pursue:

> Since both repos are open sourced, I can fork them (which I have) and add print statements/modify code to debug as needed.
> 
> I am running into a couple issues that I'm trying to resolve. I don't want you to suggest any code yet, let's think this through.
> 
> 1. When performing retrieval on a 2.6M document collection on an RTX3090, RAGatouille.search throws an OOM error.
> 2. So I chose to run retrieval on the RAGatouille index using vanilla ColBERT and it did not run out of memory.
> 3. However, the retrieval results are *significantly* different between ColBERT and RAGatouille.
> 
> Each of these gives me a uniquely interesting direction to pursue:
> 
> 1. Why does RAGatouille throw the OOM error? 2.6M documents (index with 8.5GB disk space) is not small, but not terribly large. There's an issue open in RAGatouille where they note that changing batch_size in score_pids in IndexScorer resolves an OOM error during search. I want to give this a try!
> 2. Why does ColBERT not run out of memory? But RAGatouille does?
> 3. Why are the retrieval results between RAGatouille and ColBERT different? The RAGatouille documentation says the following, which leads me to believe they should yield the same results:
> 
> If you'd like to use more than RAGatouille, ColBERT has a growing number of integrations, and they all fully support models trained or fine-tuned with RAGatouille! The official ColBERT implementation has a built-in query server (using Flask), which you can easily query via API requests and does support indexes generated with RAGatouille! This should be enough for most small applications, so long as you can persist the index on disk.
> 
> Each of these explorations are fascinating, and I think I'm going to pursue each one.
> 
> 1. resolving the RAGatouille OOM error would solve my immediate problem. ideally I tackle this first.
> 2. Understanding memory usage between RAGatouille and ColBERT has been an ongoing interest of mine. I have memory profiled both before during indexing, but not during search. This would be a very interesting research task.
> 3. Debugging the searching/scoring difference would be probably the hardest task. I would likely have to trace down all function calls, checking intermedite values, comparing them between the two frameworks. Absolutely fascinating and would learn a ton. Would also be a significant achievement to resolve the discrepancy (maybe something in the Config? Maybe something more fundamental?)

TBD on whether I pursue points 2 and 3.
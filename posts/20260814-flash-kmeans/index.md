---
title: Integrating flash-kmeans into the ColBERT repo (initial setup)
date: "2026-08-14"
author: Vishal Bakshi
description: I'm dusting off my ColBERT maintenance hat and getting back into the rhythm of things...
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

I'm dusting off my ColBERT maintenance hat and getting back into the rhythm of things.

The first thing I'm working on for the next release is exploring the most likely integration of flash-kmeans ([arxiv](https://arxiv.org/pdf/2603.09229), [github](https://github.com/svg-project/flash-kmeans), [pypi](https://pypi.org/project/flash-kmeans/)) as a replacement for faiss-gpu.

flash-kmeans (2026 Yang, et al)  is a ridiculously fast “IO-aware batched K-Means clustering implemented with Triton GPU kernels.” 

I'll be digging more into its internals later this fall when I read the paper and do a deeper dive into their repo.

In this blog post I'm going to recap my experience debugging some transformers and CUDA errors during a basic flash-kmeans integration into the ColBERT repo.

## Replacing faiss-gpu with flash-kmeans

This was pretty trivial to do (a near identical copy/paste of the existing FAISS implementation, thank you Yang and team!):

```python
def compute_flash_kmeans(dim, num_partitions, kmeans_niters, shared_lists, return_value_queue=None):
    from flash_kmeans import batch_kmeans_Euclid # ta-da!

    sample = shared_lists[0][0]  # Extract sample (same as FAISS path)
    sample_cuda = sample.cuda() 
    sample_batched = sample_cuda.unsqueeze(0)
    torch.save(sample_batched, f"{ROOT}/flash_input_sample.pt")

    cluster_ids, centers, _ = batch_kmeans_Euclid(
        sample_batched,
        n_clusters=num_partitions,
        max_iters=kmeans_niters,  
        verbose=True
    )
    torch.save(centers, f"{ROOT}/flash_raw_centers.pt")
    centroids = centers.squeeze(0)  # (1, K, D) → (K, D)
    centroids = centroids.float().cpu()  # Match FAISS output: float32 on CPU

    print_memory_stats(f'RANK:0*')

    if return_value_queue is not None:
        return_value_queue.put(centroids)

    return centroids
...

if KMEANS_ALGO == "flash": # for debugging
    print("USING FLASH KMEANS ===========================") # for debugging
    centroids = compute_flash_kmeans(*args_)
else:
    centroids = compute_faiss_kmeans(*args_)
```

## Modal Installs Wrong flash-kmeans Version

I'm not really sure how or why this happened. I'm just chalking it up as “weird things that happen in a docker image.” Older versions of flash-kmeans did not allow for an embedding dimension that wasn’t a power of 2\. Shuo Yang fixed that recently\! So as long as you install 0.3.1 you won’t get this error:

\`\`\`  
triton.compiler.errors.CompilationError: at 39:13:  
    """  
    pid\_n \= tl.program\_id(0)          \# tile index along N dimension  
    pid\_b \= tl.program\_id(1)          \# batch index  
    n\_start \= pid\_n \* BLOCK\_N  
    n\_offsets \= n\_start \+ tl.arange(0, BLOCK\_N)  
    n\_mask \= n\_offsets \< N  
    \# \------------------------------------------------------------------  
    \# Load x tile  (BLOCK\_N, D)  
    \# \------------------------------------------------------------------  
    offs\_d \= tl.arange(0, D)  
             ^  
arange's range must be a power of 2  
\`\`\`

## `transformers==4.57.0` is **yanked**

Don’t use it.

## ColBERT's JIT-compiled CUDA extension `decompress_residuals_cpp` fails to build

I feel like I get this error every year and then forget how I resolved it, so hopefully this will be the last time I forget it.

First: I was erroneously installing both `cuda -c nvidia/label/11.7.1` and `torch==2.10.0` which is not compatible with it.  
Second: Once I installed it was clashing with gcc 14, giving me a `ninja: build stopped: subcommand failed.` error so I had to pin it \<14:

\`\`\`  
micromamba create \-n colbert python=3.11 cuda-toolkit "gxx\_linux-64\<14" \-c nvidia/label/cuda-12.6.0 \-c conda-forge  
\`\`\`

Third: I was getting a ninja build error which needed the following as [advised by contributor Robin Narsingh Ranabhat](https://github.com/stanford-futuredata/ColBERT/issues/371#issuecomment-3251906773) and modified a bit by Opus 4.6 to fit my Dockerfile

\`\`\`  
ENV CONDA\_DEFAULT\_ENV=colbert  
ENV PATH=/opt/conda/envs/colbert/bin:$PATH  
ENV CONDA\_PREFIX=/opt/conda/envs/colbert  
ENV CC=$CONDA\_PREFIX/bin/x86\_64-conda-linux-gnu-gcc  
ENV CXX=$CONDA\_PREFIX/bin/x86\_64-conda-linux-gnu-g++  
ENV CUDA\_HOST\_COMPILER=$CONDA\_PREFIX/bin/x86\_64-conda-linux-gnu-g++  
\`\`\`

## AttributeError: 'HF\_ColBERT' object has no attribute 'all\_tied\_weights\_keys'

transformers defines all\_tied\_weights\_keys in the [PretrainedModel.post\_init() method](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/modeling_utils.py#L1388).  [jkaniewski-tii’s comment](https://github.com/huggingface/transformers/issues/42832#issuecomment-3971880082) on this very informative and helpful issue thread advised the reader to call `self.post_init()` in the model’s init. I added that to hf\_colbert.py’s HFColBERT init and it resolved this issue\!

## Ghost indentation error

After adding that the first couple of times, I rebuilt the image in the modal, and I got an indentation error in the file. The third time I deployed the app, I didn't, so, shrug.

## Next: analyze flash-kmeans artifacts during indexing\!

My absolute favorite part of ColBERT Maintenance, and really any project, is following the data through the pipeline, in this case, the indexing pipeline. Every time any data goes through a transformation, I save the artifact so I can inspect it later. This process will probably take at least a week, and I'll be posting updates on Twitter (@vishal\_learner) as it happens, along with another blog post. Happy maintaining\!


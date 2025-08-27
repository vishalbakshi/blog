---
title: PyTorch Version Impact on ColBERT Index Artifacts&#58; 2.4.1 --> 2.5.0
date: "2025-08-26"
author: Vishal Bakshi
description: Analysis of ColBERT indexing differences between PyTorch 2.4.1 and 2.5.0. The root cause is floating point divergence in BERT's intermediate linear layer under mixed precision with small batch sizes.
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

## Background

In a previous blog post I outlined two things:

1. Which two subsequent PyTorch versions caused a divergence in stanford-futuredata/ColBERT index `.pt` artifacts (ConditionalQA document collection):

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

2. That the difference in ColBERT index artifacts between `torch==1.13.1` and `torch==2.1.0` was a result of floating point precision divergence during the forward pass of the underlying `BertModel`'s 10 encoder layers, maximum absolute difference between each PyTorch version's layers' outputs:

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

In this blog post I'm going to show that the difference in ColBERT indexes between `torch==2.4.1` and `torch==2.5.0` is due to <mark>mixed precision forward pass divergence in the `BertModel` for small batch sizes</mark>.

## `torch==2.4.1` vs `torch==2.5.0` Index Artifact Comparison

Similar to the difference between `torch==1.13.1` and `torch==2.1.0`, most artifacts don't match between 2.4.1 and 2.5.0:

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

Also similar to 1.13.1 vs 2.1.0, swapping `local_sample_embs` resolves all intermediate artifact differences:

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

## Inspecting `batches`

In 1.13.1 vs 2.1.0, all embeddings in generated when encoding documents were different between versions, this was explained by the divergence in `BertModel` per-layer outputs. For 2.4.1 vs 2.5.0, only the _last batch of embeddings_ were different between versions. The first 31 batches of embeddings had shape `[32, 71, 96]` (batch size x max seq len x emb dim), the last batch had shape `[8, 71, 96]`. This was the first "smell" about where the problem was. These embeddings, `batches`, are generated with the following code in `colbert/modeling/checkpoint.py`:

```python
batches = [
    self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
    for input_ids, attention_mask in tqdm(
        text_batches, disable=not showprogress
    )
]
```

`checkpoint.doc` was the method of interest:

```python
def doc(self, *args, to_cpu=False, **kw_args):
    with torch.no_grad():
        with self.amp_manager.context():
            D = super().doc(*args, **kw_args)

            if to_cpu:
                return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

            return D
```

Here's the super class' `.doc` method, `ColBERT.doc`:

```python
def doc(self, input_ids, attention_mask, keep_dims=True):
    assert keep_dims in [True, False, 'return_mask']

    input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
    D = self.bert(input_ids, attention_mask=attention_mask)[0]
    D = self.linear(D)
    mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
    D = D * mask

    D = torch.nn.functional.normalize(D, p=2, dim=2)
    if self.use_gpu:
        D = D.half()

    if keep_dims is False:
        D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
        D = [d[mask[idx]] for idx, d in enumerate(D)]

    elif keep_dims == 'return_mask':
        return D, mask.bool()

    return D
```

## Mixed Precision `BertModel` Forward Pass Divergence

I found that the similarity of intermediate artifacts generated in `checkpoint.doc` between PyTorch versions depended on floating point precision.

Here's a table showing the different artifacts of different precision types I compared between `torch==2.4.1` and `torch.2.5.0`:

|Artifact|Precision|Batch Size|`torch.allclose`|
|:-:|:-:|:-:|:-:|
|Per-Layer `BertModel` Outputs|Full|32|`True`|
|`checkpoint.bert(input_ids, attention_mask=attention_mask)[0]`|Full|32|`True`
|`checkpoint.linear(D)`|Full|32|`True`
|`torch.nn.functional.normalize(D, p=2, dim=2)`|Full|32|`True`
|Per-Layer `BertModel` Outputs|Full|8|`True`|
|`checkpoint.bert(input_ids, attention_mask=attention_mask)[0]`|Full|8|`True`
|`checkpoint.linear(D)`|Full|8|`True`
|`torch.nn.functional.normalize(D, p=2, dim=2)`|Full|8|`True`
|Per-Layer `BertModel` Outputs|Mixed|32|`True`|
|`checkpoint.bert(input_ids, attention_mask=attention_mask)[0]`|Mixed|32|`True`
|`checkpoint.linear(D)`|Mixed|32|`True`
|`torch.nn.functional.normalize(D, p=2, dim=2)`|Mixed|32|`True`
|Per-Layer `BertModel` Outputs|Mixed|8|`False`|
|`checkpoint.bert(input_ids, attention_mask=attention_mask)[0]`|Mixed|8|`False`
|`checkpoint.linear(D)`|Mixed|8|`False`
|`torch.nn.functional.normalize(D, p=2, dim=2)`|Mixed|8|`False`


Mixed precision (`with torch.cuda.amp.autocast():`) alone was not sufficient to cause divergence. When combining mixed precision with a batch size of 8, the floating point values diverge. Why? The intermediate linear layer (384 --> 1536) appears to be the source of divergence for the batch-size of 8 + mixed precision divergence across PyTorch versions. Note that it didn't matter which 8-items were selected (from the first or last batch, or in between), this divergence took place between PyTorch versions.

To isolate what in `checkpoint.bert` was causing this divergence, I replaced different `checkpoint.bert` modules with `Identity`, defined as:

```python
class Identity(torch.nn.Module):
    def forward(self, x):
        return x
```

Ultimately I landed on the following code, replacing two of the dense layers with `Identity`:

```python
for layer in checkpoint.bert.encoder.layer:
    layer.intermediate.dense = Identity()
    layer.output.dense = Identity()
```

After running the scripts with this model modification, mixed precision 8-item batches yielded identical results across PyTorch versions!

|Artifact|Precision|Batch Size|`torch.allclose`|
|:-:|:-:|:-:|:-:|
|Per-Layer `BertModel` Outputs|Mixed|8|`True`|
|`checkpoint.bert(input_ids, attention_mask=attention_mask)[0]`|Mixed|8|`True`
|`checkpoint.linear(D)`|Mixed|8|`True`
|`torch.nn.functional.normalize(D, p=2, dim=2)`|Mixed|8|`True`

Here are the two modules in question: (`layer.intermediate.dense` and `layer.output.dense`)

```python
(intermediate): BertIntermediate(
    (dense): Linear(in_features=384, out_features=1536, bias=True)
    (intermediate_act_fn): GELUActivation()
)
(output): BertOutput(
    (dense): Linear(in_features=1536, out_features=384, bias=True)
    (LayerNorm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
)
```

Running the following small reproduction of the two linear layers:

```python
layer = checkpoint.bert.encoder.layer[0]
x32 = torch.randn(32, 71, 384).cuda()
x8 = x32[:8]

with torch.cuda.amp.autocast():
    out32 = layer.intermediate.dense(x32) 
    out8 = layer.intermediate.dense(x8)

print(f"Intermediate Linear match: {torch.allclose(out32[:8], out8)}")

x32_wide = torch.randn(32, 71, 1536).cuda()
x8_wide = x32_wide[:8]

with torch.cuda.amp.autocast():
    out32 = layer.output.dense(x32_wide)
    out8 = layer.output.dense(x8_wide)

print(f"Output Linear match: {torch.allclose(out32[:8], out8)}")
```

Prints out the following:

```
Intermediate Linear match: False
Output Linear match: True
```

The intermediate layer (projecting from 384 to 1536 dimensions) causes the divergence in floating point values between the first 8 items of a batch of 32 and all items in the batch of 8 for the same PyTorch version (`2.4.1`). It's interesting that the largest matrix multiplication is causing this divergence. 

Additionally, this divergence between intermediate dense layer outputs of the first n-items of a batch size of 32 and a smaller batch size of n exists for n = 8, 9 and 10, as checked by the following code:

```python
layer = checkpoint.bert.encoder.layer[0]
x32 = torch.randn(32, 71, 384).cuda()

for i in range(32):
    xs = x32[:i]
    
    with torch.cuda.amp.autocast():
        out32 = layer.intermediate.dense(x32) 
        outs = layer.intermediate.dense(xs)
    
    print(f"{i} Intermediate Linear match: {torch.allclose(out32[:i], outs)}")
```

```
...
5 Intermediate Linear match: True
6 Intermediate Linear match: True
7 Intermediate Linear match: True
8 Intermediate Linear match: False
9 Intermediate Linear match: False
10 Intermediate Linear match: False
11 Intermediate Linear match: True
12 Intermediate Linear match: True
...
```

## Appendix: Code


Here's the core functionality that I used to generate and save full precision `BertModel` (and related) artifacts:

```python
text_batches, reverse_indices = torch.load(f'{MOUNT}/{project}/{date}-{source}-{nranks}/tensorize_output.pt')
input_ids = text_batches[0][0][:8]
attention_mask = text_batches[0][1][:8]

outputs_dict = {}
def capture_output(name):
    def hook_fn(module, input, output):
        outputs_dict[name] = output[0].detach()
    return hook_fn

hooks = []
for i in range(10): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
for h in hooks: h.remove()

D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
D = checkpoint.linear(D)
mask = torch.tensor(checkpoint.mask(input_ids, skiplist=checkpoint.skiplist), device=checkpoint.device).unsqueeze(2).float()
D = D * mask
D = torch.nn.functional.normalize(D, p=2, dim=2)
```

For mixed precision I indented everything after a `with torch.cuda.amp.autocast():` line.

My code to compare two versions' artifacts generally looked like this:

```python
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
console = Console(force_terminal=True)

a = torch.load(f"{root_a}/outputs_dict.pt")
b = torch.load(f"{root_b}/outputs_dict.pt")

for i in range(10):
    a_ = a[f"{i}"]
    b_ = b[f"{i}"]
    console.print(f"Layer {i}", torch.allclose(a_, b_))

def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
def _compare(fn):
    a = torch.load(f"{root_a}/{fn}")
    b = torch.load(f"{root_b}/{fn}")
    console.print(_print(f"{fn} torch.allclose:", torch.allclose(a, b), True))

_compare("D_bert.pt")
_compare("D_linear.pt")
_compare("D_mask.pt")
_compare("D_norm.pt")
```
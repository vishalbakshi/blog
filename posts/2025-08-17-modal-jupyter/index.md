---
title: TIL&#58; Launching Jupyter with a Custom Modal Image and Volume
date: "2025-08-17"
author: Vishal Bakshi
description: I can now use Jupyter Notebooks with full Modal image/volume functionality. This unlocks a ton of productivity gains!
filters:
   - lightbox
lightbox: auto
categories:
    - ColBERT
---

Yesterday I learned of the Modal docs example showing how to start [a jupyter server via a Modal tunnel](https://github.com/modal-labs/modal-examples/blob/main/11_notebooks/jupyter_inside_modal.py). I was elated to see this because it solved my problem of not being able to specify a custom image when using `modal launch jupyter`. 

I have a Dockerfile which installs `colbert-ai` from the `main` branch of the stanford-futuredata/ColBERT repo with a specific PyTorch and Transformers version:

```Dockerfile
FROM mambaorg/micromamba:latest

USER root

RUN apt-get update && apt-get install -y git nano curl wget build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/stanford-futuredata/ColBERT.git /ColBERT && \
    cd /ColBERT && \
    micromamba create -n colbert python=3.11 cuda -c nvidia/label/11.7.1 -c conda-forge && \
    micromamba install -n colbert faiss-gpu -c pytorch -c conda-forge && \
    micromamba run -n colbert pip install -e . && \
    micromamba run -n colbert pip install torch==1.13.1 transformers==4.38.2 pandas

ENV CONDA_DEFAULT_ENV=colbert
ENV PATH=/opt/conda/envs/colbert/bin:$PATH

WORKDIR /

RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
    echo "micromamba activate colbert" >> ~/.bashrc

CMD ["/bin/bash"]
```

I then modified the Modal documentation example as follows (`jupyter_inside_modal.py`) to use my Dockerfile to create an image and use an existing Modal Volume:


```python
import subprocess
import time
import modal
from modal import Image, App, Secret, Volume
import datetime
import os

SOURCE = os.environ.get("SOURCE", "")
VOLUME = Volume.from_name("colbert-maintenance", create_if_missing=True)
MOUNT = "/colbert-maintenance"
image = Image.from_dockerfile(f"Dockerfile.{SOURCE}", gpu="L4")

app = App("jupyter-tunnel", image=image.pip_install("jupyter"))
JUPYTER_TOKEN = "" # some list of characters you'll enter when accessing the Modal tunnel

@app.function(max_containers=1, volumes={MOUNT: VOLUME}, timeout=10_000, gpu="L4")
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 10_000):
    run_jupyter.remote(timeout=timeout)
```

I then run the following locally form my terminal:

```
SOURCE="0.2.22.main.torch.1.13.1" modal run jupyter_inside_modal.py
```

Where my Dockerfile is in the same folder as `jupyter_inside_modal.py` and titled `Dockerfile.0.2.22.main.torch.1.13.1`. I can then access the cloned repo as well as my mounted volume and use a Jupyter Notebook to explore data, iterate on function definitions, compare model weights, add hooks to ColBERT models, and so on. This unlocks a ton of productivity and iteration velocity that I was scratching my head on how to obtain without the use of notebooks.
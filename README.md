# Anti-Exploration by Random Network Distillation

This repository contains an official implementation of
[Anti-Exploration by Random Network Distillation](https://arxiv.org/abs/2301.13616). All code is written in Jax.

## Dependencies & Docker setup
To set up python environment (with dev-tools of your taste, in our workflow we use conda and python 3.8), 
just install all the requirements:

```commandline
python install -r requirements.txt
```

However, in this setup, you would also need to install mujoco210 binaries by hand. Sometimes this is not super straightforward,
but we used this recipe:
```commandline
mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
```
You may also need to install additional dependencies for mujoco_py. 
We recommend following the official guide from [mujoco_py](https://github.com/openai/mujoco-py).

### Docker

We also provide a simpler way, with a dockerfile that is already set up to work, all you have to do is build and run it :)
```commandline
docker build -t sac_rnd .
```
To run, mount current directory:
```commandline
docker run -it \
    --gpus=all \
    --rm \
    --volume "<PATH_TO_THE_REPO>/sac-rnd-jax:/workspace/sac-rnd-jax" \
    --name sac_rnd \
    sac_rnd bash
```
## How to reproduce experiments

Configs for the main experiments are stored in the `configs/sac-rnd/<task_type>`. All available hyperparameters are listed in the  `offline_sac/algorithms/<algo>.py`.

For example, to start SAC-RND training process with `halfcheetah-medium-v2` dataset, run the following:
```commandline
python offline_sac/algorithms/sac_rnd.py \
    --config_path="configs/sac-rnd/halfcheetah/halfcheetah_medium.yaml" \
    --beta=<take the best value from the paper appendix>
```

To reproduce our sweeps, create wandb sweep from configs in `configs/sweeps`. After that, start wandb agent with created sweep ID. That's all! Have fun!

# Citing
If you use this code for your research, please consider the following bibtex:
```
@article{nikulin2023anti,
  title={Anti-Exploration by Random Network Distillation},
  author={Nikulin, Alexander and Kurenkov, Vladislav and Tarasov, Denis and Kolesnikov, Sergey},
  journal={arXiv preprint arXiv:2301.13616},
  year={2023}
}
```
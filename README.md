# A [Malmo](https://github.com/Microsoft/malmo) reinforcement learning environment

![Visdom graphs](https://i.imgur.com/85mSFlY.png)

## Project structure
* Agents - These set of classes control the environment. They define the task, and given a command from a policy - pass it to Malmo and return a tuple (s, a, r, t).
* Policies - These define the behavior of the agent. Can be either hard-coded, user controlled (for example, the interactive policy) or machine learned.
* Utilities:
  * Parallel agents wrapper - Allows a single policy to control multiple agents at once. Instead of running multiple distributed agents using asyncronous policies, all these agents play given the same policy but due to the stochastic nature (epsilon greedy exploration and random initialization spots) we will receive more diverse trajectories.
  * Helpers - Provides some helper functions such as plotting a line graph using visdom or playing a full episode (i.e from start upto timeout/death/successful finish).
  * Replay memory - Contains 2 configurations
    * Prioritized experience replay - Sampling from the experience replay (ER) is based on the sample priority. The priority is set to be the abs(TD-error) so as to give higher priority to samples on which we have a larger error.
    * Success experience replay - Creates a parallel, but smaller, ER. This ER will store only trajectories which lead to a successful finish of the task. In a case where finishing is a very rare event, this ensures the agent doesn't 'forget'.

### Notes
* Each policy must have a corresponding parameter file with the same name. This allows the parameters to be automatically loaded on runtime, given each policy has a unique set of parameters.
* Two new signals are available that do not exist in previous works.
  * Success - since the problems in Minecraft have a defined goal, unlike ATARI, we can leverage this signal for better and faster learning (you can decide not to use this, and disable it via the Agent interface).
  * Timeout - to ensure the agent isn't stuck in an infinite loop and to help with better exploration, we terminate after T timesteps. This termination, for most policies, is a non-markovian signal. Since this is meant to help learning and not damage it, the replay memory will not sample tuples `(s, a, r, t, s')` where `t` is a termination signal due to timeout.

### Policies out of the box
This package comes with several prebuilt policies:
* Interactive - Allows user controlled behavior over the environment using `{'w','s','a','d','space','e'}` and `'q'` which will quit.
* [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [C51-DQN - Distributional](https://arxiv.org/abs/1707.06887)

![C51-DQN](https://i.imgur.com/UHZWnOl.png "Probability distribution function.")
* [QR-DQN - Quantile Regression](https://arxiv.org/abs/1710.10044)

![QR-DQN](https://i.imgur.com/spMScJs.png "Cumulative distribution function.")

### Requirements
* Python 3.6
* [PyTorch](http://pytorch.org/) version 0.3 - for built in policies.
* [Visdom](https://github.com/facebookresearch/visdom) (if you want to view plots online).
* [Malmo](https://github.com/Microsoft/malmo)

### Running the agent
```
python3.6 main.py <policy> <agent> [parameters]
```

For instance, to automatically load Malmo:
```
python3.6 main.py qr-dqn single_room --number_of_atoms 200 --number_of_agents 1 --retain_rgb --save_name qr-dqn-test
```

A more robust solution is to open Malmo externally and provide the ports (default port for Malmo is 10000):
```
python3.6 main.py qr-dqn single_room --number_of_atoms 200 --number_of_agents 1 --malmo_ports 10000 --retain_rgb --save_name qr-dqn-test
```

### Disclaimer
This codebase was built for my research, this is not an official product of any sort.
I urge users to submit issues, bug reports and fixes to help make this better for everyone.

To cite this repository in publications:
```
@misc{malmo_rl,
  author = {Chen Tessler},
  title = {Malmo Reinforcement Learning Environment},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tesslerc/malmo_rl/}},
}
```

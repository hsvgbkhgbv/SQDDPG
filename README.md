# Rethink Global Reward Game and Credit Assignment in Multi-agent Reinforcement Learning

This project implements the algorithm of Shapley Q-value policy gradient (SQPG) and demonstrates the experiments in comparison with Independent DDPG, Independent Actor-critic, MADDPG and COMA.

The corresponding paper is: 

All of algorithms are implemented in Python (3.5.4), with Pytorch (1.0), so please install the relevant dependencies before running the codes or invoking the functions.

The suggested solution is to install Anaconda Python (3.5.4) version: https://www.anaconda.com/download/.
To enable the environments, please install OpenAI Gym (0.10.5) and Numpy (1.14.5).
After installing the related dependencies mentioned above, open the terminal and execute the following bash script:
```bash
cd multi-agent-rl/environments/multiagent_particle_envs/
pip install -e .
```

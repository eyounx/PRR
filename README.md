# PRR
PRR code for the paper
> WenJi Zhou, Yang Yu, Yingfeng Chen, Kai Guan, Tangjie Lv, Changjie Fan, Zhi-Hua Zhou. **Reinforcement Learning Experience Reuse with Policy Residual Representation**. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI'19), Macao, China.


## Requirement
- python 3.6
- rllab
- numpy == 1.16
- tensorflow==1.8.0
- gym==0.13.0
- scipy==1.3.0
- matplotlib==3.1.1
- theano==1.0.4
- cached_property==1.5.1

## Files

- main.py   is the PRR demo on three FetchTheKey environments in the paper
- envs      contains test environment
- nets      network structures (modified from https://github.com/pat-coady/trpo)
- rl        contains agent and policy codes

## Run Demo
- To run experiment on FetchTheKey environment, just running:
```bash
python main.py
#Actor-Critic with Synthesis Loss
---
Pytorch implementation of Actor-Critic with Synthesis Loss. Algorithm is tested on MuJoCo continuous tasks version3 in OpenAI gym. Networks are trained using PyTorch 1.8.1 and Python 3.7

##Usage

`python main.py --env-name <environment_name> --cuda --limit_kl --kl_target 5e-3 --agentname <name>`


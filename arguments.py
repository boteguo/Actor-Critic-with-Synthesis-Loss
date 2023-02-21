import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Error controlled Actor-Critic Args')
    parser.add_argument('--env', default="HalfCheetah-v3") 
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G') 
    parser.add_argument('--tau', type=float, default=0.005, metavar='G')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='G')
    parser.add_argument('--kl_target', type=float, default=0.5, metavar='G')
    parser.add_argument('--reward_scale', type=float, default=5, metavar='G') 
    parser.add_argument('--batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N')
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N')
    parser.add_argument('--Q_updates_per_step', type=int, default=1, metavar='N')
    parser.add_argument('--max_episode_steps', type=int, default=1000, metavar='N')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N')
    parser.add_argument('--memory_size', type=int, default=500000, metavar='N')
    parser.add_argument('--update_interval', type=int, default=2, metavar='N') 
    parser.add_argument('--eval_interval', type=int, default=5000, metavar='N') 
    parser.add_argument('--limit_kl', action="store_true")
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--GPU-id', type=int, default=0, metavar='N')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agentname',default="ACSL")
    parser.add_argument('--eta',type = float,default=1)
    parser.add_argument('--special',default = "")
    parser.add_argument('--expscalar',type = float,default=0.2)
    parser.add_argument('--save_model',action = "store_true")
    parser.add_argument('--clamp_q',action = "store_true")
    parser.add_argument('--softmax',action = 'store_true')
    parser.add_argument("--mode",type = str, default="exp")
    parser.add_argument("--max_lambda",type = float, default=float("inf"))

    args = parser.parse_args()
   
    args.cuda = args.cuda and torch.cuda.is_available()

    return args

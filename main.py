import argparse
import os
import pybullet_envs
import gym
import mujoco_py
import numpy as np
import itertools
import torch
import threading
import csv
import random
from datetime import datetime
from env import make_pytorch_env
import ACSL
from arguments import get_args


def run(args):

    if args.cuda:
        torch.cuda.set_device(args.GPU_id)
        print('Program is running on GPU {}'.format(args.GPU_id))

    env = make_pytorch_env(args.env, clip_rewards=False)
    test_env = make_pytorch_env(
        args.env, episode_life=False, clip_rewards=False)

    SEED = args.seed
    env.seed(SEED)
    env.action_space.seed(SEED)
    test_env.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


    try:
        max_episode_steps = env._max_episode_steps
    except:
        max_episode_steps = args.max_episode_steps

    env_beta_map ={
        "Ant-v3":0.001,
        "Walker2d-v3":0.1,
        "Hopper-v3":0.05,
        "HalfCheetah-v3":0.005,
        "Humanoid-v3":0.05
    }
    beta = env_beta_map[args.env] if args.env in env_beta_map else 0.001
    
    path = f"result/{args.agentname}_{args.env}_{args.seed}{args.special}"
    agent = ACSL.ACSLAgent(env, test_env,path,args.batch_size, args.lr, 
                     memory_size=args.memory_size, 
                     hidden_dim=args.hidden_dim, gamma=args.gamma, tau=args.tau, 
                     limit_kl=args.limit_kl, kl_target=args.kl_target,
                     update_interval = args.update_interval, 
                     Q_updates_per_step=args.Q_updates_per_step, 
                     num_steps=args.num_steps, max_episode_steps=max_episode_steps, 
                     cuda=args.cuda,args = args,beta=beta)

    episode_reward = 0 # Total reward in one episode.
    episode_steps = 0 # Total steps in one episode.
    episode_num = 0
    state,done = env.reset(),False 

    R,R_Q = agent.evaluate()
    evaluations = [R]
    minerq = [R_Q]
    Q = []
    r = 0
    if args.clamp_q:
        log_lambd = torch.zeros(1,requires_grad=True,device = agent.device)
        lambd_optim = torch.optim.Adam([log_lambd],lr = 0.003) if args.mode == "exp" else torch.optim.Adam([log_lambd],lr = 0.01) #relu
        delta = torch.zeros(1,requires_grad=True,device = agent.device) if args.mode == "exp" else torch.ones(1,requires_grad=True,device = agent.device)
        delta_optim = torch.optim.Adam([delta],lr = 0.003)
    else:
        log_lambd = -1
        delta = -1
    total_numsteps = 0 # Counting total actions steps.
    debug = False
    for i in range(args.num_steps):
        total_numsteps += 1

        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample action from uniform distibution.
        else:
            action = agent.explore(state)  # Sample action from policy.
        Q.append(agent.Q(state,action)[0])


        next_state, reward, done, _ = env.step(action) # Interact with env.
        episode_steps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        mask = 1 if episode_steps == max_episode_steps else float(not done) 
        # Append the transition to memory
        scaled_reward = reward * args.reward_scale
        agent.memory.push(state, action, scaled_reward, next_state, mask)
        state = next_state
        r += reward
        # Update all parameters of the networks.
        if len(agent.memory) > args.batch_size and total_numsteps >= args.start_steps:
            agent.update_parameters(log_lambd,args.expscalar,delta,debug)
            debug = False
        if done:
            if args.clamp_q:
                if total_numsteps > args.start_steps:
                    r_q = ((1-agent.gamma)*episode_steps*np.mean(Q) + agent.gamma * Q[0])/args.reward_scale
                    r_q = torch.tensor(r_q).to(agent.device)
                    r = torch.tensor(r * args.eta).to(torch.device("cuda"))
                    if r > 0:
                        lambd_optim.zero_grad()
                        loss = (r - r_q) * log_lambd
                        loss.backward()
                        lambd_optim.step()
                        lambd_optim.zero_grad()

                        delta_optim.zero_grad()
                        loss2 = (r - r_q) * delta
                        loss2.backward()
                        delta_optim.step()
                        delta_optim.zero_grad()
                    #print(f"R:{r.item()}, R_Q: {r_q.item()}")
                    #print("lambd = ",np.exp(torch.clamp(log_lambd,max=3).item())*args.expscalar, "delta = ",(torch.sigmoid(delta)/4 +0.75).item())
                    #print("lambd = ",(torch.exp(log_lambd)*args.expscalar).item(), "delta = ",(torch.sigmoid(delta)/2 +0.5).item())
            episode_num += 1
            print(f"total_numsteps = {total_numsteps},episode num = {episode_num}, return = {episode_reward}")
            episode_reward = 0 # Total reward in one episode.
            episode_steps = 0 # Total steps in one episode.
            done = False # Episode end flag.
            state = env.reset() # Reset the env.
            Q = []
            r = 0  
        
        if total_numsteps % args.eval_interval == 0:
            debug = True
            R,R_Q = agent.evaluate()
            evaluations.append(R)
            minerq.append(R_Q)
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(f"{path}/R",evaluations)
            np.save(f"{path}/R_Q",minerq)
            if args.save_model:
                model_dir = os.path.join(path, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                agent.save_models(model_dir)



    env.close()
    test_env.close()
    

if __name__ == "__main__":
    args = get_args()
    run(args)

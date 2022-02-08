for ((i=0;i<10;i+=1))
do
    python main.py --env Ant-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode exp \
               --expscalar 0.01

    python main.py --env Walker2d-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode exp \
               --expscalar 0.01
    
    python main.py --env HalfCheetah-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode relu \
               --expscalar 0.2
    
    python main.py --env Humanoid-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode relu \
               --expscalar 0.2 \
               --reward_scale 20
    
    python main.py --env Hopper-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
    
    python main.py --env Swimmer-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode relu \
               --expscalar 0.2
    
    python main.py --env Reacher-v2 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode relu \
               --expscalar 0.2
    
    python main.py --env HumanoidStandup-v2 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --clamp_q \
               --mode relu \
               --expscalar 0.2 \
               --rewrad_scale 1
    
done
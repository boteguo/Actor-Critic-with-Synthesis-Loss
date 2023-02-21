for ((i=0;i<10;i+=1))
do
python main.py --env Ant-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode exp \
               --expscalar 0.01 \
               --num_steps 3000001 \
               --agentname ACSL \
               --max_lambda 0.5 \
               --clamp_q 

python main.py --env HalfCheetah-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode relu \
               --expscalar 0.2 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q 

python main.py --env Hopper-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode exp \
               --expscalar 0.01 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q \
               --max_lambda 0.5 \

python main.py --env Humanoid-v3 \
               --seed $i \
               --cuda \
               --GPU-id 0 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode relu \
               --expscalar 0.2 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q \
               --mode relu \
               --reward_scale 20

python main.py --env HumanoidStandup-v2 \
               --seed $i \
               --cuda \
               --GPU-id 1 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode exp \
               --expscalar 0.01 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q \
               --mode exp \
               --reward_scale 1

python main.py --env Reacher-v2 \
               --seed $i \
               --cuda \
               --GPU-id 2 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode exp \
               --expscalar 0.01 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q 

python main.py --env Swimmer-v3 \
               --seed $i \
               --cuda \
               --GPU-id 2 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode exp \
               --expscalar 0.01 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q 

python main.py --env Walker2d-v3 \
               --seed 0 \
               --cuda \
               --GPU-id 3 \
               --save_model \
               --limit_kl \
               --kl_target 5e-3 \
               --mode exp \
               --expscalar 0.01 \
               --num_steps 3000001 \
               --agentname ACSL \
               --clamp_q 

done
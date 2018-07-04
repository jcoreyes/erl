#!/bin/bash

python visualization/grill/reacher_baselines.py
python visualization/grill/pusher_baselines.py
python visualization/grill/multiobj_pusher_baselines.py

python visualization/grill/reacher_online_ablation.py
python visualization/grill/pusher_online_ablation.py

python visualization/grill/reacher_reward_type_ablation.py
python visualization/grill/pusher_reward_type_ablation.py
python visualization/grill/multiobj_pusher_reward_type_ablation.py

python visualization/grill/reacher_relabeling_ablation.py
python visualization/grill/pusher_relabeling_ablation.py
python visualization/grill/multiobj_pusher_relabeling_ablation.py

python visualization/grill/real_reacher.py

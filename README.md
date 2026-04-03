# Forward-Looking Decision Mamba: Offline RL with Contrastive Prefix-Equivalence

This is the official implementation of the paper **Forward-Looking Decision Mamba**.



##  Installation
Our environment dependencies are consistent with the original https://github.com/Toshihiro-Ota/decision-mamba. 

1. Clone this repo:
   ```bash
   git clone https://github.com/huiyinglun51-creator/FL-DMamba.git
   cd FL-DMamba
##  Dependencies   
   ```bash
conda create -n FL-Mamba python=3.9
conda activate FL-Mamba
pip install -r requirements.txt

   
##  Quick start
   ```bash
python experiment.py  --env 'hopper'         --dataset 'medium'                 --K 20         --model_type 'dmamba'         --n_layer 3         --embed_dim 256         --activation_function 'gelu'         --max_iters 10         --batch_size 64   --num_steps_per_iter 10000         --learning_rate 1e-4         --weight_decay 1e-4         --num_eval_episodes 100             --seed 0

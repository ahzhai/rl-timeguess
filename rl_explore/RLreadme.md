this is the command to run for the rl path:

python -m rl_explore.rl_train \               
  --metadata data/gsv-cities/Dataframes \
  --data_root data/gsv-cities \
  --samples_per_city 50

Run 1:

  python -m rl_explore.rl_train \
  --metadata data/gsv-cities/Dataframes \
  --data_root data/gsv-cities \
  --samples_per_city 80 \
  --epochs 15 \
  --steps_per_batch 2048 \
  --baseline_checkpoint checkpoints/baseline_cnn.pt

  Results --> 
  Running final test evaluation ...
  Final test (2.3s) — higher is better:
  Median GeoGuessr score: 457.35
  Mean GeoGuessr score:   808.98



Run 2 (with attention):

python -m rl_explore.rl_train \
  --metadata data/gsv-cities/Dataframes \
  --data_root data/gsv-cities \
  --samples_per_city 100 \
  --epochs 20 \
  --steps_per_batch 2048 \
  --baseline_checkpoint checkpoints/baseline_cnn.pt \
  --step_penalty 0.025 \
  --ent_coef 0.02 \
  --coord_coef 0.7 \
  --lr 2e-4 \
  --max_steps 10 \
  --ppo_epochs 4 \
  --minibatch_size 256

Results --> 
Running final test evaluation ...
Final test (4.4s) — higher is better:
  Median GeoGuessr score: 991.46
  Mean GeoGuessr score:   1521.79
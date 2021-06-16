#!/bin/bash

# We do not recommend running the script as is, but to use it as a template for running particular scripts.

# Generate values for critic training.
python q_and_v.py --cost_smoothing 0
python q_and_v.py --cost_smoothing 0.5

# Train critics.
python train_point_critics.py --cost_smoothing 0
python train_point_critics.py --cost_smoothing 0.5

# Iterate over seeds
for s in 0 10 20 30 40 50 60 70 80 90; do

  # Primal-dual optimization (PDO)
  ./cppo.py --epochs 500 --optimize_penalty -s $s --exp_name point_pdo

  # PDO with shaped cost
  ./cppo.py --epochs 500 --optimize_penalty -s $s --cost_smoothing 0.5 \
    --exp_name point_pdo_shaped

  # Intervention with point critic (sparse cost)
  ./cppo.py --epochs 500 --ignore_unsafe_cost --intv_config intv/critic_no_smoothing.yaml -s $s \
    --exp_name point_critic_intv_sparse_cost

  # Intervention with point critic (shaped cost)
  ./cppo.py --epochs 500 --ignore_unsafe_cost --intv_config intv/critic_smoothing.yaml -s $s \
    --exp_name point_critic_intv_shaped_cost

  # Intervention with true model and sparse cost
  ./cppo.py --epochs 500 --ignore_unsafe_cost --intv_config intv/sparse.yaml \
    -s $s --exp_name point_intv_true_model_sparse_cost

  # Intervention with biased model and sparse cost
  ./cppo.py --epochs 500 --ignore_unsafe_cost --intv_config intv/sparse.yaml \
    --model_config model/small.yaml -s $s --exp_name point_intv_biased_model_sparse_cost

  # Intervention with true model and shaped cost
  ./cppo.py --epochs 500 --ignore_unsafe_cost --intv_config intv/shaped.yaml \
    -s $s --exp_name point_intv_true_model_shaped_cost

  # Intervention with biased model and shaped cost
  ./cppo.py --epochs 500 --ignore_unsafe_cost --intv_config intv/shaped.yaml \
    --model_config model/small.yaml -s $s --exp_name point_intv_biased_model_shaped_cost

  # CSC (sparse cost)
  ./csc.py --epochs 500 --optimize_penalty -s $s --alpha 0. --exp_name point_csc_sparse_cost

  # CSC (shaped cost)
  ./csc.py --epochs 500 --optimize_penalty -s $s --cost_smoothing 0.5 --exp_name point_csc_shaped_cost
done

#!/bin/bash

# We do not recommend running the script as is, but to use it as a template for running particular scripts.

# Iterate over seeds
for s in 0 10 20 30 40 50 60 70; do

  # Primal-dual optimization (PDO)
  ./cppo.py --epochs 1250 --optimize_penalty -s $s --exp_name cheetah_pdo

  # PDO with shaped cost
  ./cppo.py --epochs 1250 --optimize_penalty -s $s --cost_smoothing 0.05 \
    --exp_name cheetah_pdo_shaped

  # Heuristic-based intervention with tightened height range
  ./cppo.py --epochs 1250 --ignore_unsafe_cost --intv_config intv/heuristic.yaml --heuristic_intv \
    -s $s --exp_name cheetah_heuristic

  # Heuristic-based intervention with original height range
  ./cppo.py --epochs 1250 --ignore_unsafe_cost --intv_config intv/heuristic_original.yaml --heuristic_intv \
    -s $s --exp_name cheetah_heuristic_original

  # MPC-based intervention with unbiased model (24 hours per seed!!!)
  ./cppo.py --epochs 1250 --ignore_unsafe_cost --intv_config intv/mppi_unbiased.yaml \
    -s $s --exp_name cheetah_mpc_unbiased

  # MPC-based intervention with biased model (24 hours per seed!!!)
  ./cppo.py --epochs 1250 --ignore_unsafe_cost --intv_config intv/mppi_biased.yaml \
    -s $s --exp_name cheetah_mpc_biased

  # CSC (sparse cost)
  ./csc.py --epochs 1250 --optimize_penalty --alpha 0.1 \
    --exp_name cheetah_csc_sparse -s $s

  # CSC (shaped cost)
  ./csc.py --epochs 1250 --optimize_penalty --cost_smoothing 0.05 --alpha 0.1 \
    --exp_name cheetah_csc_shaped -s $s
done

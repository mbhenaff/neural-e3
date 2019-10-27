## Explicit Explore-Exploit Algorithms in Continuous State Spaces

All code was written with Python 3.6 and PyTorch v1.1.0.

To run Neural-E3 on continuous control tasks, do:

```
cd src/
python train_e3.py -env {mountaincar, acrobot} -n_exploration_epochs 10 -n_trajectories 10 -eps 0.1 -n_model_updates 200 -n_exploration_epochs 10 
```

To run it on maze tasks, do:

```
cd src/
python train_e3.py -env {maze5, maze10, maze20} -n_exploration_epochs 10  -lambda_r 0.01
```

The combination lock code is organized a little differently because it builds on the codebase of Du et al.
To run Neural-E3 on that domain, do:

```
cd combolock/src/
python Experiment.py --alg neural-e3 --horizon 5 --lr 0.01 --n_playouts 200 --n_ensemble 5 --seed 1 --env_param_1 0.1 
```








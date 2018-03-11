# Deep-reinforcement-learning

This repository contains the code for various deep reinforcement learning algorithms such as Policy Gradients and Deep Q-Learning.

## Q-learning

Training an agent using deep Q learning to navigate the given game environment. There are two game environments that the Q-learning agent can be trained on.

* *Gridworld*

A simple grid like arena where the agent must navigate to the end tile, carefully avoiding pitfalls. The agent is awarded more points the faster it reaches the end. 

We use the bellman equation as the state space is discrete.

The agent currently converges to the correct path within a few minutes of trials.

In order to see the agent in action, run
```
python3 Learner.py
```


* *MountainCar*

Here, the agent must control a car in order to move it up the slope of a hill. The agent must learn to gain momentum by first moving backwards and only then move forward in order to reach the final goal. 

Unlike gridworld, the position of the car is a continuous variable, so we use a neural network as a learning agent in order to output the correct actions.

The agent learns how to climb the hill within a few minutes of training.

In order to see the agent in action, run
```
python3 Q_learning.py
```



## Policy Gradients

* *Pong* 

A classic atari game. The agent learns to move the paddle in order to defeat the AI opponent. The agent only takes the raw pixels of the game as an input which is fed to a neural net which outputs the appropriate actions. 

Actions which lead to losses are discouraged and actions which lead to wins are encouraged.(Through gradient ascent/descent)

The tensorflow version has been trained for over a day on a CPU, and shows great improvement over the initial random agent.

In order to train the agent, run
```
python3 policy_grad.py
```
For the tensorflow version(Training on GPU is recommended), run
```
python3 policy_grad_tf.py
```

### Credits 

Credits go to Siraj Raval for the starter code.

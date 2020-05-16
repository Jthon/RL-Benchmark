# RL-Benchmark
Implemention of several benchmark reinforcement learning algorithm
### Vanilla Policy Gradient
VPG is a simple policy gradient algorithm,in this repo, I trained the model in both discrete 
action space enviroment(CartPole-V1,LunarLander-v2) and continuous action space(not implemented yet). <br>
##### Psudocode:
<center>![Vanilla Policy Gradient](https://github.com/Jthon/RL-Benchmark/blob/master/Vanilla_PG/result/vpg.jpg)</center>

##### Experiment:
||curve|result|
---|:--:|:--:|
CartPole-v1|![cartpole_curve](https://github.com/Jthon/RL-Benchmark/blob/master/Vanilla_PG/result/CartPole-v1/curve.jpg)|![cartpole_result](https://github.com/Jthon/RL-Benchmark/blob/master/Vanilla_PG/result/CartPole-v1/epi%3D1000.gif)|
LunarLander-v2|![lunarlander_curve](https://github.com/Jthon/RL-Benchmark/blob/master/Vanilla_PG/result/LunarLander-v2/curve.jpg)|![lunarlander_result](https://github.com/Jthon/RL-Benchmark/blob/master/Vanilla_PG/result/LunarLander-v2/epi%3D8900.gif)|




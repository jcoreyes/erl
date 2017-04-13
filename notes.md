# TODOs
 - TODO's in code
 - Make the network init stuff its own class/namedtuple
 - Perform/set up mechanism for ablation tests
 - Make a load_policy method
 - Organize the misc folder
 - See TODO in test_tensorflow.py
 - Have my own trajectory/path/episode class/abstraction that interacts well
 with the "paths" that rllab has
 - Add a way to profile stuff run on EC2. Right now, the prof file isn't saved.
 - Check if there's a bug in the static_rnn and dynamic_rnn code. It 
 seems that their call_cell lambda should pass in the scope, but maybe that's
  just me

## ICML
 - Add a version of DDPG where the policy outputs a distribution over discrete actions
 - Save figures of bptt doing worse on horizon of 100
 - Why is mem state DDPG unstable?

# Notes
These notes are really for myself (vpong), so they're probably meaningless to anyone else.
I just push them so that they're backed up.

# Ideas
 - Do Laplace smooth for OnehotSampler
 - Decay exploration noise.

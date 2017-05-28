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

## NIPS 2017
Still runnning: HighLow
- 5-17-benchmark-our-method-full-bptt-H32-hl/
- 5-17-benchmark-our-method-no-bptt-H32-hl/
- 5-17-dev-benchmark-our-method-hl/

Still runnning: WaterMaze
- 5-17-benchmark-our-method-full-bptt-watermaze-easy-2/
- 5-17-benchmark-our-method-full-bptt-watermaze-2/
- 5-17-dev-our-method-water-maze-easy/

Still running, benchmarks: WaterMaze
- 5-17-benchmark-mtrpo-watermaze-batchsize10000
- 5-17-policy-type-mddpg-watermaze-easy/
    - keep for mddpg baseline



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

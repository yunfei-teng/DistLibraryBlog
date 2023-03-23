# Discussion

#### Hyperparameter Selection

For learning rates and pulling strengths, in a relatively reasonalble range, smaller values usually encourage lower final test error while larger values encourage faster convergence.

For communication period, this depends on the network condition. The data transfer speed can be checked by netowrk tool `iperf`.

#### Batch Normaliztion
One dilemma for distributed training is to deal with running mean and running variance of [batch normalization](https://arxiv.org/abs/1502.03167). Based on our knowledge, there is no perfect solution to synchronize running mean and running variance for asynchronous communication so far. Usually, when synchronizing we can ignore running mean and running variance so that each model (worker) keeps their own statistics. In our implementation, we tried both (1) averaging both running mean and running variance of all workers for every commumication period and (2) ignoring them (As a side note, for PyTorch 1.0 version, synchronizing the empty tensors will cause an error, so make sure to comment out the parts of synchronizing the buffers).

#### Feedback
The codes were developed when PyTorch first officially released its distributed training library (PyTorch 1.0 version). Feedbacks and questions are welcome!

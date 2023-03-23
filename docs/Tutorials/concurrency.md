# Concurrency

## **1. Concurrency**
### (1.a) MultiThreading
* CPUs are added for increasing computing power.
* Many processes are executed **simultaneously**.
* Every process owned a separate address space.
### (1.b) MultiProcessing
* Many threads are created of a single process for increasing computing power.
* Many threads of a process are executed **sequentially** due to *Global Interpretation Lock (GIL)*.
* All threads shared a common address space.

### (1.c) Daemon Thread [(Reference)](https://docs.python.org/2/library/threading.html)
A thread can be flagged as a “daemon thread”. The significance of this flag is that the entire Python program exits when only daemon threads are left. The initial value is inherited from the creating thread. The flag can be set through the daemon property.

!!! warning
    Daemon threads are abruptly stopped at shutdown. Their resources (such as open files, database transactions, etc.) may not be released properly. If you want your threads to stop gracefully, make them non-daemonic and use a suitable signalling mechanism such as an `Event`.

There is a **“main thread”** object; this corresponds to the **initial thread** of control in the Python program. It is not a daemon thread. Its initial value is inherited from the creating thread; the main thread is not a daemon thread and therefore all threads created in the main thread default to daemon = False.

### What does **join()** function in multithreading do? [[Reference]](https://stackoverflow.com/questions/15085348/what-is-the-use-of-join-in-python-threading)

!!! warning
    The following two methods have different behaviors:

    **Parallel one:**
    ```python
    task1.start()
    task2.start()
    task1.join()
    taks2.join()
    ```
    **Sequetial one**
    ```python
    task1.start()
    task1.join()
    task2.start()
    taks2.join()
    ```
    `join()` has to be done after every thread has been started.

## **2. NVIDIA Packages**
### (2.a) CORE
* PyTorch Distributed Training: [[Data Parallel Example]](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py)
* PyTorch Distributed Training Operations: [[distributed_c10]](https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py)
* PyTorch NCCL: [[NCCL Library]](https://github.com/pytorch/pytorch/blob/792cb774f152bab5b968f4ec51da0fc21ff9e895/torch/lib/c10d/ProcessGroupNCCL.cpp)
* CUDA Events: [[CUDA Event]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
* CUDA Semantics: [[CUDA Semantics]](https://pytorch.org/docs/stable/notes/cuda.html)

### (2.b) CUDA Events
* SynChronous Case
    ``` python
    '''Waiting on the work's corresponding CUDA events'''
    void ProcessGroupNCCL::WorkNCCL::synchronize() {
      for (size_t i = 0; i < devices_.size(); ++i) {
        auto currentStream = at::cuda::getCurrentCUDAStream(devices_[i].index());
        '''Block the current stream on the NCCL stream'''
        cudaEvents_[i].block(currentStream);
        '''If we use the work to do barrier, we should block here'''
        if (!barrierTensors_.empty()) {
          at::cuda::CUDAGuard gpuGuard(devices_[i]);
          AT_CUDA_CHECK(cudaDeviceSynchronize());
        }
      }
    }
    ```

* Asynchronous Case
    ``` python3
    bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
      for (size_t i = 0; i < devices_.size(); ++i) {
        '''Checking the work's corresponding CUDA events' status'''
        auto ret = cudaEventQuery(cudaEvents_[i]);
        if (ret != cudaSuccess && ret != cudaErrorNotReady) {
          AT_CUDA_CHECK(ret);
        }
        if (ret == cudaErrorNotReady) {
          return false;
        }
      }
      return true;
    }
    ```
!!! help
    * If NCCL doesnot work, try to use MPI and disable NCCL [[cuda aware mpi]](https://devblogs.nvidia.com/introduction-cuda-aware-mpi/).
    * `nvidia-smi` and `cuda device` inconsistency. (https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150)

## **3. Discussion**
### Potential Issues to avoid
* Using *multithreading* for synchronous experiments as all the processes can share the memory with each other. However, for asynchronous experiments this will surely not be appreciated due the the low efficiency.

* Be careful about concurrency of Python as this script language is tricky. When there is some serious bug it may not report anything but pretend everything is OK.

* Should use `watch -n 0.1 nvidia-smi` to monitor the performance of GPUs. *Loading data* usually takes much more time than expected.

* Should not call `torch.cuda.*` before spwaning subprocesses. e.g. `torch.cuda.device_count()`. The explantion is right here [(report useful error)](https://github.com/pytorch/pytorch/issues/15734).

* "Each subprocess should be assingned an individual GPU to make use of NCCL." This is the suggestion proporsed in this issue (https://github.com/pytorch/pytorch/issues/15051). However, based on my test even if spawning 4 processes on the same GPU NCCL still works fine.

* TCP/IP port racing problem was fixed in PyTorch 1.0.1 version. Before that version, running distribution initialization on **master** machine is mandatory.

### Debugging
#### Distributed Training Operation Tests
Run `distributed_training_test()` and check the results of all tests.
``` python
# define waiting function 
def dumb_wait(req, name = "asynchronization"):
    print("Entering " + name)
    start = time.time()
    while(not req.is_completed()):
        pass
    async_time = time.time() - start
    print(name + " spent {async_time:.5f} seconds".format(async_time=async_time))

# define necessary things for testing 
def distributed_training_test():
    # Synchronous distribution training test
    print("\n[Synchronous distribution testing...]")

    torch.distributed.broadcast(tensor, src=0)
    print(" -Passed broadcast test")

    torch.distributed.all_reduce(tensor)
    print(" -Passed all_reduce test")

    dist.all_gather(tensor_list, tensor)
    print(" -Passed all_gather test")

    torch.distributed.broadcast_multigpu([tensor], src=0)
    print(" -Passed broadcast_multigpu test")

    torch.distributed.all_reduce_multigpu([tensor])
    print(" -Passed all_reduce_multigpu test")

    dist.all_gather_multigpu([tensor_list], [tensor])
    print(" -Passed all_gather_multigpu test")

    # Asynchronous distribution training test
    print("\n[ASynchronous distribution testing...]")
    req = torch.distributed.broadcast(tensor, src=0, async_op=True)
    dumb_wait(req)
    print(" -Passed broadcast test")

    torch.distributed.all_reduce(tensor, async_op=True)
    dumb_wait(req)
    print(" -Passed all_reduce test")

    dist.all_gather(tensor_list, tensor, async_op=True)
    dumb_wait(req)
    print(" -Passed all_gather test")

    torch.distributed.broadcast_multigpu([tensor], src=0, async_op=True)
    dumb_wait(req)
    print(" -Passed broadcast_multigpu test")

    torch.distributed.all_reduce_multigpu([tensor], async_op=True)
    dumb_wait(req)
    print(" -Passed all_reduce_multigpu test")

    dist.all_gather_multigpu([tensor_list], [tensor], async_op=True)
    dumb_wait(req)
    print(" -Passed all_gather_multigpu test")
```

!!! bug "GPU parallelism bug"
    For some specific GPUs, there might be some problem of parallelism and see this [discussion](https://github.com/pytorch/pytorch/issues/1637#issuecomment-338268158) for solution.
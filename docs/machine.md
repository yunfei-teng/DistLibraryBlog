# Distributed Training Environment
## Local Setup
On this webpage one should be able to find related information about both hardware and software setups for distriuted training.
!!! info "Limitation"
    In this tutorial we only consider training nodes equiped with NVIDIA GPUs.

### 1. Hardware configuration
The performance of hardware has a strong influence on the training speed as expected.

1. **GPUs** Make sure the remote servers have available Nvidia GPUs (GTX 1080, RTX 2080, RTX 3008, etc.). These computational processors are key to training the deep learning models faster.

2. **CPUs** Type `head /proc/cpuinfo` to check CPU information. Though most computation cost should be given to GPUs, the performance of CPU (like the number of cores) has strong influence on some multi-processing tasks like loading the traning and testing data. Besides, the choice of number of workers for data loader depends on the number of CPU cores .

3. **Storage Device** Check the information of the storage device where the training and testing data is stored. Usually solid-state drive (SSD) has much faster reading and loadind speed than hard disk drive (HDD).

### 2. Software environment
To properly configure the software envirionment used for distributed training, the following steps are necessary to be done:

#### (1) NVIDIA driver
Depending on the CUDA version, the Nvidia driver has to be properly installed. Take a look at Nvidia website for [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).
```bash
apt search nvidia-driver              # search resources to install Nvidia driver
sudo apt install nvidia-driver-460    # install latest version of Nvidia driver
sudo reboot                           # reboot the computer
```
!!! note
    Usually, installing the latest version of NVIDIA driver is the best choice. The reason is that CUDA of higher versions are usually only supported by lastest NVIDIA driver.

After installation and rebooting, type the following commands in the terminal.
```bash
nvidia-smi                 # check current status of GPUs on the machine
watch -n0.1 nvidia-smi     # check status of GPUs on the machine for every 0.1 second
```
If the returned information looks like then the GPUs and NVIDAI driver work functionally.
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.73.01    Driver Version: 460.73.01    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:09:00.0 Off |                  N/A |
| 27%   29C    P8     5W / 180W |      2MiB /  8119MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 00000000:0A:00.0 Off |                  N/A |
| 28%   28C    P8     6W / 180W |      2MiB /  8119MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 1080    Off  | 00000000:41:00.0 Off |                  N/A |
| 28%   32C    P8     5W / 180W |      2MiB /  8119MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 1080    Off  | 00000000:42:00.0 Off |                  N/A |
| 28%   33C    P8     6W / 180W |     17MiB /  8117MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
!!! help
    1. On the top row of the table, if shows the both the versions of NVIDIA driver and CUDA library.
    2. Besides the information of NVIDIA driver and CUDA library, the table also indicated the number of GPUs the operating system *could* recognize and the status of GPUs (e.g. temperature, memory, utility, etc.).
    3. Usually we are cared about two columns: Memort-Usage and GPU-Util. If the memory consumption exceeds the capacity of current GPU the GPU will stop working directly. While if the value of GPU-Util remains freeezed then the GPU is not working as expected. 
    4. NVIDIA driver will be installed at the root repository so different users share the same driver.

#### (2) PyTorch
Go to the official website of [PyTorch](https://pytorch.org/) and install the corresponding PyTorch version.
!!! note
    1. CUDA library will be implicitly installed along with PyTorch.
    2. PyTorch will be installed in each user's local home folder so different users will have their own PyTorch installed.

## Remote Connection
### (1) SSH to the server
To do distributed training, we usually create and modify the codes on a local machine first and then talk with the GPU servers through `ssh`.
!["ssh to the server"](ssh.png)
#### (1.a) Create new user
Use the following command to create a new user if necessary. Different users will usually share the same NVIDIA driver but different PyTorch versions.
```bash
sudo adduser [username]
sudo usermod -aG sudo [username]
```
#### (1.b) Password-free login
On local machine, generage amn SSH key first
```bash
ssh-keygen
```
Then copy the key to a destinate server
```bash
ssh-copy-id user@hostname.example.com
```
After these two steps one should be able to directly login into the remote server without entering passwords. However, when logining from the server to the local machine a password is still needed, so if one would like to login from the servers back to the local machine without being asked for a password just repeat the same steps on the server side.
!!! help
    1. Make sure the user name is consistent with the one on the local machine if one would like to login without typing in the username explicitly. This is helpful when there are multiple users on the same machine.
    2. How to `ssh` a machine by customized name instead of server's ip address? On Ubuntu, do `vim /etc/hosts` in the terminal and add a new line like `[server_ip_address] [my_server]`. After refreshing one could direcly do `ssh [my_server]` to login into the remote server.

### (2) Test bandwidth between servers
Install `iperf` package which is included in most Linux distribution’s repositories.
```bash
apt-get install iperf
```
On the terminal of one server
```bash
iperf -s
```
On the terminal of the other server
```bash
iperf -c [first_server_ip_address]
```
The returned results shoud look like
```bash
------------------------------------------------------------
Server listening on TCP port 5001
TCP window size:  128 KByte (default)
------------------------------------------------------------
[  4] local 128.238.9.108 port 5001 connected with 128.238.9.109 port 42488
[ ID] Interval       Transfer     Bandwidth
[  4]  0.0-10.0 sec  6.58 GBytes  5.65 Gbits/sec
```
!!! note "Network bandwidth"
    The Bandwidth is indicated in **Gbits/sec** where `1 Byte = 8 bits`. The common Ethernet usually supports a bandwidth of `1 Gbits/sec` which might be too slow for distriuted training. In order to improve the performance for distributed training, a minimum bandwidth of `10 Gbits/sec` is necessary. That requires the upgrade of network cable, network switch and network interface controller. The data center might use InfiniBand directly which is much faster and more expensive as well.

### (3) Excecute remote command
* Execute remote command
```bash
ssh [my_server] '[command]'
```
For example, one might always do `ssh [my_server] 'pkill -9 python'` to kill all python processes on the remote server for distributed training in case the training pipeline breaks.

* Run multiple commands
```bash
ssh [my_server] '[command1]; [command2]; [command3]'
```
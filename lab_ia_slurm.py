import os
from subprocess import call

# n[1-5]:
#     2 x Intel Xeon E5-2620 v4 8 cores / 16 threads @ 2.40 GHz (Haswell)
#     128 GiB of RAM
#     2 x NVIDIA RTX A6000 with 48GiB of GDDR6 (PCIe)
# n[51-55]:
#     2 x Intel Xeon Gold 5120 14 cores / 28 threads @ 2.2GHz (Skylake)
#     192 GiB of RAM
#     3 x NVIDIA Tesla V100 with 32 GiB of RAM (PCIe)
# n[101-102]:
#     2 x Intel Xeon Gold 6148 20 cores / 40 threads @ 2.4 GHz (Skylake)
#     384 GiB of RAM
#     4 x NVIDIA Tesla V100 with 32 GiB of RAM (NVLink)

_LAB_IA_NODES = [
    ["n"+str(i) for i in range(1, 6)],
    ["n"+str(i) for i in range(51, 56)],
    ["n"+str(i) for i in range(101, 103)]
]

_LAB_IA_CPUCORE_PER_GPU = {
    "n[1-5]": 8,
    "n[51-55]": 4, # 4.666667 in reality
    "n[101-102]": 10
}

_LAB_IA_QOS = [
    "default", # This QoS allows a user to run up to 6 jobs with up to 6 GPU for up to 24 hours. Jobs running on this QoS are uninterruptible, meaning that requested resources will be assign to a user for the duration of the jobs. If the jobs exceed 24 hours, Slurm will kill all its process to reclaim the resources. If a job ends earlier, the resources are freed.
    "preempt", # This QoS works the same way that default does. The only difference is that jobs running on preempt are interruptible. If someone runs a job on default or testing, it might stop a job running on preempt. This partition is intented to run extra jobs when Lab-IA is underused.
    "debug", # This QoS allows a user to run 1 job with up to 2 GPU for up to 30 minutes. It is intented for testing purposes only. Please use this QoS if you need to test that a job can run on a node before running it on other partitions.
    "nvlink", # This QoS allows a user to run a single job with up to 4 GPU on the pcie partition.
    "pcie" # This Qos allows a user to run a single job with up to 4 GPU on the nvlink partition.
]

_LAB_ID_PARTITIONS = [
    "all", # This is the default parition. It allows any user to access every nodes.
    "testing", # This is the testing partition. It allows any user to test his code on every types of nodes.
    "pcie", # This is an exclusive partition. It allows a user to access every resources on a single node (CPU and memory) where GPU are connected with PCI Express. This partition must be used if a job needs to run multi-GPU jobs. Since using this partition will prevent any other user to access the node, please use it wisely.
    "nvlink", # Same comment as PCIE
]

_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-task={ncpus}
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --output=slurm/log/{exp_name}.out
#SBATCH --error=slurm/log/{exp_name}.err
#SBATCH --time={time_max}
#SBATCH --job-name={exp_name}
{slurm_addon}

{cmd} 
{script_addon}

"""


def gpu_jobs_submitter(
        cmd:str,
        exp_name:str,
        ngpus:int=1,
        qos:str="default",
        partition:str="all",
        time_max:str="24:00:00",
        slurm_addon:str="",
        script_addon:str="",
    ):
    # sanity checks
    assert qos in _LAB_IA_QOS, f"qos must be one of {_LAB_IA_QOS}"
    assert partition in _LAB_ID_PARTITIONS, f"partition must be one of {_LAB_ID_PARTITIONS}"
    assert ngpus >= 0, "ngpus must be >= 0"

    # create slurm dir if not exists
    if not os.path.exists("slurm/log"):
        os.makedirs("slurm/log")

    # fill sbatch script
    filled_sbatch = _SLURM_TEMPLATE.format(
        ngpus=ngpus,
        ncpus=4, # best to modify so that it fits the ratio depending on node selected
        qos=qos,
        time_max=time_max,
        partition=partition,
        exp_name=exp_name,
        slurm_addon=slurm_addon,
        script_addon=script_addon,
        cmd=cmd
    )
    slurm_script_path = os.path.join(f"slurm/{exp_name}.slurm")
    open(slurm_script_path, 'w').write(filled_sbatch)        
    call(['sbatch', slurm_script_path])
    print(f"{ngpus} GPUs / 4 cpus per task")
        


import random
import json

# EXAMPLE KEY: S_3-Ord_10-I_0-WIP_0

# read all the instances in-memory
with open("thesis_benchmarks.json", 'r') as file:
    instances = json.load(file)
print(len(instances))

# 50x6, 20x9, 100x3

wanted_sizes = [("30", "6"), ("10", "9"), ("50", "3")]

out_file_name = "instance_sample.json"

instance_sizes = set([instance_name.split("-I_")[0] for instance_name in instances.keys()])
available_trials = len(instances) // len(instance_sizes)
wip_constant_range = 0

print(f"Assuming {available_trials} trials")

instance_sample = {}
for instance_name in instance_sizes:
    instance_stages = instance_name.split("S_")[1].split("-")[0]
    instance_jobs = instance_name.split("-Ord_")[1].split("-")[0]

    if (instance_jobs, instance_stages) not in wanted_sizes:
        print(f"{(instance_jobs, instance_stages)} skipped.")
        continue

    print(f"Choosing instance for size {instance_name}")
    random_trial = random.randint(0, available_trials - 1)
    instance_sample[instance_name] = instances[f"{instance_name}-I_{random_trial}-WIP_{wip_constant_range}"]

with open(out_file_name, 'w') as file:
    json.dump(instance_sample, file, indent=4)
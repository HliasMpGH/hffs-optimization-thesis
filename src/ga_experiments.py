from opt.operator.crossover import Crossover
import json
from util.objectives import Objective
import os

import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
from opt.tuning import GATuner

from util.ffs_preprocessing import data_preprocessing_identical


with open("input/thesis_benchmarks.json", 'r') as file:
    data = json.load(file)

with open("input/tests_eaai.json", 'r') as file:
    tests = json.load(file)

os.makedirs("visuals", exist_ok=True)

ga_tests = tests["GA"]

instance_results = {}

optuna_tune = ga_tests["experiments"]["optuna_use"]
trials_num = ga_tests["experiments"]["trials"]

print(f"read input. Executing GA Experiments: Trials={trials_num}, Optuna={optuna_tune}")
# iterate over the instances
for instance_number, instance in data.items():

    # the results for each method for this instance
    method_results = {"GA": {}}

    # preprocess the instance
    general, stages, process, \
    job_stage_elig, stage_job_elig, \
    transTimes, orders, jobs, WIPs, \
    cells_per_stage = data_preprocessing_identical(instance)

    # contain the problem attributes
    problem_attributes = {
        "orders": orders,
        "stages": stages,
        "job_list": jobs,
        "job_stage_elig": job_stage_elig,
        "process": process,
        "transTimes": transTimes,
        "WIPs": WIPs
    }

    order_names = list(orders.keys())

    # change order-X to just X, as pyGAD requires numbers for the genes
    orders_only_numbers = [int(order_name.split("-")[1]) for order_name in order_names]

    # define the fitness function, derived by the makespan
    makespan_fitness = Objective(problem_attributes).makespan_fitness


    # define a GATuner for the instance
    ga_tuner = GATuner(
        problem_attributes=problem_attributes,
        **ga_tests["tune"],
        crossover_operator=None, # dont define an operator to pass one later
        valid_genes=orders_only_numbers,
        fitness_function=makespan_fitness,
        n_trials=trials_num, # run each combination multiple times for more accurate results
        time_limit=90 * 60 # 90 minutes time limit for each GA run
    )

    # define a crossover instance to retrieve the operators
    crossover_instance = Crossover(problem_attributes)

    # use all available operators
    for crossover_name in ga_tests["crossover_operators"]:
        print(f"tuning operator: {crossover_name}")

        # define the crossover operator
        crossover_operator = crossover_instance.get_by_name(crossover_name)

        # tune the parameters of the GA
        if optuna_tune:
            # find the best combination using an optuna search
            print("tuning by optuna")
            optuna_study, combination_results = ga_tuner.tune(crossover_operator=crossover_operator, optuna_trials=1)

            # save results for the best run of each operator
            method_results["GA"][crossover_name] = {
                "detailed_trial_results": combination_results,
                "best_results": {
                    "best_params": ga_tuner.best_params,
                    "chromosome": list(ga_tuner.best_chromosome),
                    "fitness": ga_tuner.best_fitness,
                    "generation_found": int(ga_tuner.generation_found),
                    "time_found": ga_tuner.time_found,
                    "total_time": ga_tuner.total_time
                }
            }
        else:
            # find the best combination using brute force
            print("tuning brutely")
            combination_results = ga_tuner.tune_brutely(crossover_operator)

            # save best fitness plots for each instance/operator
            ga_tuner.best_ga.plot_fitness(
                save_dir=f"visuals/fitness_{crossover_name}_{instance_number}.png",
                title=f"{crossover_name} - Stages: {general['numStages']}, Orders: {general['numOrders']}"
            )

            # save results for the best run of each operator
            method_results["GA"][crossover_name] = {
                "detailed_trial_results": combination_results,
                "best_results": {
                    "best_params": ga_tuner.best_params,
                    "chromosome": list(ga_tuner.best_chromosome),
                    "fitness": ga_tuner.best_fitness,
                    "generation_found": int(ga_tuner.generation_found),
                    "time_found": ga_tuner.time_found,
                    "total_time": ga_tuner.total_time
                }
            }

    # save total instance results
    instance_results[instance_number] = {
        "benchmark": general,
        "method:": method_results
    }

print(instance_results)

with open("ga_results_eaai.json", 'w') as file:
    json.dump(instance_results, file, indent=4)

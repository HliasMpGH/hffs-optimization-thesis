import copy
import json
from opt.alns import ALNS

from util.ffs_preprocessing import data_preprocessing_identical

with open("input/thesis_benchmarks.json", 'r') as file:
    data = json.load(file)

with open("input/ga_results_eaai.json", 'r') as file:
    ga_results = json.load(file)

instance_results = {}

operator = "swap"

print(f"read input. Executing ALNS Experiments")
# iterate over the instances
for instance_number, instance in data.items():

    # the results for each method for this instance
    method_results = {"ALNS": {}}

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

    # retrieve the trial results of the GA in this instance
    instance_ga_trials = ga_results[instance_number]["method:"]["GA"][operator]["detailed_trial_results"][0]["trial_results"]

    alns_instance = ALNS(problem_attributes)

    for trial_name, ga_trial_results in instance_ga_trials.items():
        print(f"executing trial '{trial_name}'")
        # starting solution of the ALNS should be the best chromosome of first GA population
        starting_solution = [f"Order-{int(gene)}" for gene in ga_trial_results["best_starting_chromosome"]]

        # time limit of the ALNS should be the total execution time of the GA
        time_limit = ga_trial_results["total_time"]

        alns_instance.run(starting_solution, time_limit)

        method_results["ALNS"][trial_name] = {
            "parameters": {
                "starting_solution": starting_solution,
                "time_limit": time_limit
            },
            "final_operator_weights": {
                "destroy": copy.deepcopy(alns_instance.destroy_operators.operators_weights),
                "repair": copy.deepcopy(alns_instance.recreate_operators.operators_weights)
            },
            "best_cost": alns_instance.best_cost,
            "best_solution": alns_instance.best_solution,
            "iteration_found": alns_instance.iteration_found,
            "total_iterations": alns_instance.iterations_completed,
            "time_found": alns_instance.time_found,
            "total_time": alns_instance.total_time
        }

    # save total instance results
    instance_results[instance_number] = {
        "benchmark": general,
        "method:": method_results
    }

print(instance_results)

with open("alns_results.json", 'w') as file:
    json.dump(instance_results, file, indent=4)

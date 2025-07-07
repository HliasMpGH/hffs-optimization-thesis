import time
import optuna
from optuna.visualization import plot_optimization_history
from plotly.io import show
import copy
import numpy as np

from opt.operator.population_generation import PopulationGeneration

import pygad

from opt.operator.crossover import Crossover
from opt.operator.mutation import Mutation

class GATuner:

    def __init__(
            self,
            problem_attributes,
            pop_size_range,
            generations_num_range,
            selection_modes,
            crossover_operator,

            mutation_operators,
            mutation_prob_range,
            valid_genes,
            fitness_function,
            n_trials,
            time_limit=10
        ):
        self.pop_size_range = pop_size_range
        self.generations_num_range = generations_num_range
        self.selection_modes = selection_modes

        self.mutation_operators = mutation_operators
        self.mutation_prob_range = mutation_prob_range
        self.valid_genes = valid_genes
        self.fitness_function = fitness_function

        # also define crossover and mutation objects for UD operators
        self.crossover = Crossover(problem_attributes)
        self.mutation = Mutation()
        try:
            self.crossover_operator = self.crossover.get_by_name(crossover_operator)
        except ValueError:
            # provided operator is not UDF
            self.crossover_operator = crossover_operator

        self.n_trials = n_trials
        self.time_limit = time_limit

    def __on_start(self, ga_instance):
        """Called when GA starts - record start time"""
        # reset state per GA run
        self.start_time_ga = time.time()
        self.best_fitness_ga = None
        self.total_time_ga = None
        self.time_found_ga = None
        self.best_chromosome_ga = None
        self.generation_found_ga = -1
        # record the best solution of the first population to store it later
        self.best_starting_chromosome_ga = ga_instance.best_solution()[0]
        print(f"STARTING BEST: {self.best_starting_chromosome_ga}")


    def __on_generation(self, ga_instance):
        """Called after each generation - check for new global best"""
        current_time = time.time()

        # get current best fitness
        chromosome, best_fitness_ga, _ = ga_instance.best_solution()
        print(f"population best: {best_fitness_ga}({chromosome}). global best :{self.best_fitness_ga}")

        # check if this is a new global best
        if (self.best_fitness_ga is None or
            best_fitness_ga > self.best_fitness_ga):
            print(f"updated global best")
            self.best_fitness_ga = best_fitness_ga
            self.best_chromosome_ga = chromosome
            self.generation_found_ga = ga_instance.generations_completed
            self.time_found_ga = current_time - self.start_time_ga
            print(f"time FOUND: {self.time_found_ga}")

        if current_time - self.start_time_ga > self.time_limit:
            print(f"\n\n\n\nSECONDS {current_time - self.start_time_ga} -> STOPING GA\n\n\n\n")
            return "stop"

        # print("population:")
        # for i, individual in enumerate(ga_instance.population):
        #     # Calculate fitness using your fitness function
        #     fitness = ga_instance.fitness_func(ga_instance, individual, i)
        #     print(f"Individual {i}: {individual} -> Fitness: {fitness}")

    def __on_stop(self, ga_instance, fitnesses):
        """Called when GA ends - record total time"""
        self.total_time_ga = time.time() - self.start_time_ga
        # chromosome, fit, _ = ga_instance.best_solution()
        # print(f"final pygad best: {fit}({chromosome})")
        print(f"final stored best: {self.best_fitness_ga}({self.best_chromosome_ga})")

    def objective(self, trial):
        # population size tuning
        pop_size = trial.suggest_int(
            "population_size",
            self.pop_size_range[0],
            self.pop_size_range[1]
        )

        # generations number tuning
        generations = trial.suggest_int(
            "generations",
            self.generations_num_range[0],
            self.generations_num_range[1]
        )

        # parent selection tuning
        selection_mode = trial.suggest_categorical(
            "parent_selection_type",
            self.selection_modes
        )

        # mutation probability tuning
        mutation_prob = trial.suggest_float(
            "mutation_probability",
            self.mutation_prob_range[0],
            self.mutation_prob_range[1],
            step=self.mutation_prob_range[2],
        )

        # mutation operator tuning
        mutation_operator_name = trial.suggest_categorical(
            "mutation_operator_name",
            self.mutation_operators
        )

        match mutation_operator_name:
            case "local_swap":
                mutation_operator = self.mutation.local_swap

        # record the combination
        combination = {
            "pop_size": pop_size,
            "generations": generations,
            "selection_mode": selection_mode,
            "mutation_operator": mutation_operator_name,
            "mutation_prob": mutation_prob
        }

        # random population generation
        generate_random_population = PopulationGeneration(
            population_size=pop_size,
            valid_genes=self.valid_genes
        ).random

        # the fitnesses of all trials on this combination of parameters
        combination_fitnesses = []
        trial_results = {}
        for trial in range(self.n_trials):

            ga_instance = pygad.GA(
                num_generations=generations,
                num_parents_mating=2,
                keep_elitism=pop_size - 2, # keep the best 8 solutions from the previous generation (keep_parents has no effect)
                initial_population=generate_random_population(),
                mutation_type=mutation_operator,
                mutation_probability=mutation_prob,
                parent_selection_type=selection_mode,
                fitness_func=self.fitness_function,
                crossover_type=self.crossover_operator,
                on_start=self.__on_start,
                on_generation=self.__on_generation,
                on_stop=self.__on_stop
            )

            ga_instance.run()

            # store the fitness of each trial on these parameters
            combination_fitnesses.append(self.best_fitness_ga)

            print(
                f"SAVING fitness={self.best_fitness_ga}. \
                solution={list(self.best_chromosome_ga)}.\
                time_found={self.time_found_ga}.\
                total_time={self.total_time_ga}.\
                generation_found={int(self.generation_found_ga)}.\
                best_starting_chromosome={list(self.best_starting_chromosome_ga)}"
            )

            trial_results[f"trial-{trial}"] = {
                "fitness": self.best_fitness_ga,
                "time_found": self.time_found_ga,
                "total_time": self.total_time_ga,
                "chromosome": list(self.best_chromosome_ga),
                "generation_found": int(self.generation_found_ga),
                "best_starting_chromosome": list(self.best_starting_chromosome_ga)
            }

            # store the best results found
            if self.best_fitness_ga > self.best_fitness:
                print("updated best ga instance")
                self.best_fitness = self.best_fitness_ga
                self.total_time = self.total_time_ga
                self.time_found = self.time_found_ga
                self.best_chromosome = self.best_chromosome_ga
                self.generation_found = self.generation_found_ga
                self.best_params = combination

        self.combination_results.append({
            "parameters": combination,
            "trial_results": trial_results
        })

        # use the mean of trials to guide the optuna search
        return np.mean(combination_fitnesses)

    # implement logic to store the best ga_instance
    def tune(self, crossover_operator, optuna_trials=100):
        # reset each time to find best, per operator and not in total
        self.best_fitness = -1
        self.best_ga = None
        self.total_time = -1
        self.time_found = -1
        self.best_chromosome = None
        self.generation_found = -1
        self.best_params = {}

        self.combination_results = []
        # if crossover_operator is not None:
        #     print(f"Received operator to tune by optuna: {crossover_operator}")
        self.crossover_operator = crossover_operator

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=optuna_trials)

        return study, self.combination_results

    def tune_brutely(self, crossover_operator=None):
        # reset each time to find best, per operator and not in total
        self.best_fitness = -1
        self.best_ga = None
        self.total_time = -1
        self.time_found = -1
        self.best_chromosome = None
        self.generation_found = -1
        self.best_params = {}

        if crossover_operator is None:
            crossover_operator = self.crossover_operator

        # the final execution results of each combination and trial
        combination_results = []

        step = 1
        for pop_size in range(self.pop_size_range[0], self.pop_size_range[1] + 1, step):
            for generations in range(self.generations_num_range[0], self.generations_num_range[1] + 1, step):
                for selection_mode in self.selection_modes:
                    for mutation_operator_name in self.mutation_operators:
                        for mutation_prob in np.arange(self.mutation_prob_range[0], self.mutation_prob_range[1] + 0.01, self.mutation_prob_range[2]):
                            print(f"Trying parameters: (pop={pop_size}, gens={generations}, mutation_prob={mutation_prob}, selection_mode={selection_mode}, mutation_operator={mutation_operator_name})")

                            # record the combination
                            combination = {
                                "pop_size": pop_size,
                                "generations": generations,
                                "selection_mode": selection_mode,
                                "mutation_operator": mutation_operator_name,
                                "mutation_prob": mutation_prob
                            }

                            # random population generation
                            generate_random_population = PopulationGeneration(
                                population_size=pop_size,
                                valid_genes=self.valid_genes
                            ).random

                            if mutation_operator_name == "local_swap":
                                mutation_operator = Mutation(prob=mutation_prob).local_swap

                            trial_results = {}
                            # run each combination {n_trials} times for more accurate results
                            for trial in range(self.n_trials):
                                print(f"running trial {trial}")
                                # create new GA instance for every run to reset attributes
                                ga_instance = pygad.GA(
                                    num_generations=generations,
                                    num_parents_mating=2,
                                    keep_elitism=pop_size - 2,
                                    initial_population=generate_random_population(),
                                    mutation_type=mutation_operator,
                                    parent_selection_type=selection_mode,
                                    fitness_func=self.fitness_function,
                                    crossover_type=crossover_operator,
                                    on_start=self.__on_start,
                                    on_generation=self.__on_generation,
                                    on_stop=self.__on_stop
                                )

                                ga_instance.run()

                                print(
                                    f"SAVING fitness={self.best_fitness_ga}. \
                                    solution={list(self.best_chromosome_ga)}.\
                                    time_found={self.time_found_ga}.\
                                    total_time={self.total_time_ga}.\
                                    generation_found={int(self.generation_found_ga)}.\
                                    best_starting_chromosome={list(self.best_starting_chromosome_ga)}"
                                )

                                trial_results[f"trial-{trial}"] = {
                                    "fitness": self.best_fitness_ga,
                                    "time_found": self.time_found_ga,
                                    "total_time": self.total_time_ga,
                                    "chromosome": list(self.best_chromosome_ga),
                                    "generation_found": int(self.generation_found_ga),
                                    "best_starting_chromosome": list(self.best_starting_chromosome_ga)
                                }

                                if self.best_fitness_ga > self.best_fitness:
                                    print("updated best ga instance")
                                    self.best_fitness = self.best_fitness_ga
                                    self.total_time = self.total_time_ga
                                    self.time_found = self.time_found_ga
                                    self.best_chromosome = self.best_chromosome_ga
                                    self.generation_found = self.generation_found_ga
                                    self.best_params = combination
                                    self.best_ga = copy.deepcopy(ga_instance) # store to retrieve graph

                            combination_results.append({
                                "parameters": combination,
                                "trial_results": trial_results
                            })

        return combination_results
        # return self.best_params, self.best_fitness, self.total_time, self.time_found, combination_results

    def plot_history(self):
        fig = plot_optimization_history(self.study)
        show(fig)
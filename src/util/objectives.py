from util.ffs_assignment import ffs_assignment
from util.ffs_cp_modules import ffs_wip

'''
Module responsible for the definitions of objective functions.
'''

class Objective:

    valid_problem_keys = set([
        "stages",
        "job_stage_elig",
        "process",
        "transTimes",
        "orders",
        "WIPs",
        "job_list"
    ])

    def __init__(self, problem):
        # validate given problem input
        self.__validate_problem(problem)

        self.problem = problem

    '''
    Validate the given problem input.
    Each passed problem should have at least the `valid_problem_keys` attributes
    '''
    def __validate_problem(self, problem):
        given_keys = set(problem.keys())
        if self.valid_problem_keys.intersection() != self.valid_problem_keys:
            raise ValueError(
                f"Invalid Problem Input. Expected {self.valid_problem_keys} attributes but got: {given_keys}"
            )

    '''
    Compute and return the final cost of a given order sequence.
    Given a sequence, assign the jobs to machines and then
    handle the buffers before calculating the makespan.
    '''
    def makespan_fitness(self, ga_instance, sequence, sequence_idx):
            upper_cmax=10000
            stages = self.problem["stages"]
            job_stage_elig = self.problem["job_stage_elig"]
            process = self.problem["process"]
            transTimes = self.problem["transTimes"]
            orders = self.problem["orders"]
            WIPs = self.problem["WIPs"]
            job_list = self.problem["job_list"]

            C_max_no_wip, job_order, loads, starting_times, completion_times, job_cell = ffs_assignment(
                sequence, stages, process, job_stage_elig, transTimes, orders, job_list, WIPs, format_orders=True)

            C_max_wip, starting_times, completion_times, sols, cut_cp = ffs_wip(job_order, stages, process,
                                                                                            job_stage_elig, transTimes,
                                                                                            orders, job_list, WIPs,
                                                                                            job_cell,
                                                                                            C_max_upper=upper_cmax)
            #, sols, job_order, C_max_no_wip
            return 1 / C_max_wip

    def makespan_cost(self, sequence):
        upper_cmax=10000
        stages = self.problem["stages"]
        job_stage_elig = self.problem["job_stage_elig"]
        process = self.problem["process"]
        transTimes = self.problem["transTimes"]
        orders = self.problem["orders"]
        WIPs = self.problem["WIPs"]
        job_list = self.problem["job_list"]

        C_max_no_wip, job_order, loads, starting_times, completion_times, job_cell = ffs_assignment(
            sequence, stages, process, job_stage_elig, transTimes, orders, job_list, WIPs)

        C_max_wip, starting_times, completion_times, sols, cut_cp = ffs_wip(job_order, stages, process,
                                                                                        job_stage_elig, transTimes,
                                                                                        orders, job_list, WIPs,
                                                                                        job_cell,
                                                                                        C_max_upper=upper_cmax)
        #, sols, job_order, C_max_no_wip
        return C_max_wip
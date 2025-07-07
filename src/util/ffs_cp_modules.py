import numpy as np
import random
import math
from docplex.cp.model import CpoModel  # Import DOCplex for the CP Optimizer module
import cplex  # Import CPLEX solver
from cplex.exceptions import CplexSolverError
from datetime import datetime


# job_order: dict - key: machine, item: list of assigned jobs with sequence
# job_list: list of total jobs
# stages: dict : key: name of stage, item: list of machines of stage
# job_stage_elig: dict: key job, item: list of eligible stages for j
# job_cell: dict - key: job, item: list of assigned machine for j


def ffs_wip(job_order, stages, process, job_stage_elig, transTimes, orders, job_list, WIP, job_cell, C_max_upper):
    #print(datetime.now(), " - Start of CP.")
    os = "W"
    infeas_cp = 0
    C_max, sols = math.inf, {}
    timelimit = 10  # Seconds of timelimit
    printLog = False  # Print log of CP solution
    starting_times = {j: {} for j in job_list}
    completion_times = {j: {} for j in job_list}
    model = CpoModel()  # Construct a model called 'model'

    # taskVar : Interval variable for processing of job 'j' in stage 's' on machine 'm' - Fixed duration, equal to jobs[j][s]
    # waitBefore : Interval variable for waiting of job 'j', before entering machine 'm' on stage 's'
    # waitAfter : Interval variable for waiting of job 'j', after exiting machine 'm' on stage 's'
    taskVar, waitBefore, waitAfter = {}, {}, {}
    for j in job_list:
        for s in stages.keys():
            for m in stages[s]:
                if j in job_order[m]:
                    start = (0, C_max_upper)
                    end = (0, C_max_upper)
                    size = (process[s][j], C_max_upper)
                    taskVar[(j, s, m)] = model.interval_var(start, end, size, name=f"job_{j},{s},{m}")
                    waitBefore[(j, s, m)] = model.interval_var(start, end, name=f"waitBefore_{j},{s},{m}")
                    waitAfter[(j, s, m)] = model.interval_var(start, end, name=f"waitAfter_{j},{s},{m}")

    makespan = model.integer_var(0, C_max_upper, name=f'makespan')  # Integer variable for makespan

    # Constraints
    for j in job_list:
        for s in job_stage_elig[j]:
            for m in stages[s]:
                if j in job_order[m]:
                    # Processing starts at the end of waiting before entry.
                    model.add(model.start_at_end(taskVar[(j, s, m)], waitBefore[(j, s, m)]))
                    # Processing ends at the waiting after exit.
                    model.add(model.start_at_end(waitAfter[(j, s, m)], taskVar[(j, s, m)]))
                    # Makespan is greater than the completion time of any 'waiting after exit' interval
                    model.add(makespan >= model.end_of(taskVar[(j, s, m)]))

    for j in job_list:
        for s in range(len(job_stage_elig[j]) - 1):
            for m in range(len(job_cell[j]) - 1):
                if job_cell[j][m] in stages[job_stage_elig[j][s]]:
                    # Each job starts waiting to enter the machine of stage s+1 when it ends waiting to exit the machine of stage s, added by the respective travel time.

                    model.add(model.start_at_end(waitBefore[(j, job_stage_elig[j][s + 1], job_cell[j][m + 1])],
                                                 waitAfter[(j, job_stage_elig[j][s], job_cell[j][m])],
                                                 int(-transTimes[job_cell[j][m]][job_cell[j][m + 1]])))

    for s in stages.keys():
        for m in stages[s]:
            # Pulse 1: The number of entry buffers for each machine is not exceeded.
            if len(job_order[m]):
                model.add(sum(model.pulse(waitBefore[(j, s, m)], 1) for j in job_order[m]) <= WIP[m]['in'])
                # Pulse 2: The number of exit buffers for each machine is not exceeded.
                model.add(sum(model.pulse(waitAfter[(j, s, m)], 1) for j in job_order[m]) <= WIP[m]['out'])

                model.add(sum(model.pulse(taskVar[(j, s, m)], 1) for j in job_order[m]) <= 1)

            for j in range(len(job_order[m]) - 1):
                # The sequence of jobs on each machine is respected.
                model.add(
                    model.end_before_start(taskVar[(job_order[m][j], s, m)], taskVar[(job_order[m][j + 1], s, m)]))

    model.add(makespan <= C_max_upper)

    # Objective function
    model.add(model.minimize(makespan))  # Objective function: minimisation of makespan

    # If the cpoptimizer is not found, replace sol with this one:

    if os == "L":
        #sol = model.solve(agent='local',
        #                  execfile='/home/vstavros/CP/CP/ILOG/CPLEX_Studio201/cpoptimizer/bin/x86-64_linux/cpoptimizer',
        #                  TimeLimit=timelimit, SolutionLimit=1, trace_log=printLog, OptimalityTolerance=0.005)
        sol = model.solve(agent='local',
                          execfile='/home/svatikiotis/ibm_cpopt/cpoptimizer/bin/x86-64_linux/cpoptimizer',
                          TimeLimit=timelimit, SolutionLimit=1, trace_log=printLog, OptimalityTolerance=0.005)
    else:
        sol = model.solve(
            execfile='C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cpoptimizer\\bin\\x64_win64\\cpoptimizer.exe',
            TimeLimit=timelimit, trace_log=printLog, SolutionLimit=1, OptimalityTolerance=0.005)  # Solve the model

    # print(sol.get_stop_cause())
    # print(sol.get_solve_status())
    # print(sol.get_solve_time())

    if sol:  # If a feasible solution is found ...
        C_max, sols = calculate_times(sol, stages, job_stage_elig, job_cell, orders, makespan, taskVar)

    else:  # ... else ...
        C_max = math.inf
        starting_times, completion_times = math.inf, math.inf
        sols = {}
        infeas_cp = 1
        '''
        # print(f'No feasible schedule exists.')
        # print(sol.get_stop_cause())
        # print(sol.get_solve_status())
        # print(sol.get_solve_time())
        if sol.get_solve_status() != "Infeasible":
            counter = 3
            while (str(sol.get_solve_status()) != 'Feasible' and str(sol.get_solve_status()) != 'Optimal') and (
                    str(sol.get_solve_status()) != 'Infeasible') and counter < 5:

                if os == "L":
                    sol = model.solve(agent='local',
                                      execfile='/home/svatikiotis/ibm_cpopt/cpoptimizer/bin/x86-64_linux/cpoptimizer',
                                      TimeLimit=timelimit * counter, trace_log=printLog, SolutionLimit=1,
                                      OptimalityTolerance=0.005)
                else:
                    sol = model.solve(
                        execfile='C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cpoptimizer\\bin\\x64_win64\\cpoptimizer.exe',
                        TimeLimit=timelimit * counter, trace_log=printLog, SolutionLimit=1,
                        OptimalityTolerance=0.005)  # Solve the model

                print(sol.get_solve_status(), timelimit * counter, C_max_upper)
                counter += 1
            if str(sol.get_solve_status()) == 'Infeasible' or counter >= 5:
                C_max = math.inf
                starting_times, completion_times = math.inf, math.inf
                sols = {}
                infeas_cp = 1
            else:
                C_max, sols = calculate_times(sol, stages, job_stage_elig, job_cell, orders, makespan, taskVar)

            # print(sol.get_stop_cause())
            # print(sol.get_solve_status())
            # print(sol.get_solve_time())
            # print(sol.get_objective_gap())
        else:
            C_max = math.inf
            starting_times, completion_times = math.inf, math.inf
            sols = {}
            infeas_cp = 1
        '''
    #print(datetime.now(), " - End of CP.")
    return C_max, starting_times, completion_times, sols, infeas_cp


def calculate_times(sol, stages, job_stage_elig, job_cell, orders, makespan, taskVar):
    sols = {}

    C_max = sol[makespan]  # round(sol[makespan] / 60, 2)
    # print ("CP_SOL", C_max)
    for ordID in orders.keys():
        sols[ordID] = {}
        for job in orders[ordID]:
            sols[ordID][job] = {}
            # print(f"Job {job}")
            for s in job_stage_elig[job]:
                for m in stages[s]:
                    if m in job_cell[job]:
                        # print(f"Machines {m}")
                        sols[ordID][job][m] = {}
                        sols[ordID][job][m]['start'] = sol[taskVar[(job, s, m)]][
                            0]  # round(sol[taskVar[(job, s, m)]][0] / 60, 2)
                        sols[ordID][job][m]['end'] = sol[taskVar[(job, s, m)]][
                            1]  # round(sol[taskVar[(job, s, m)]][1] / 60, 2)
                        # sum_of_comp += sol[taskVar[(job, s, m)]][1]
                        # if sol[taskVar[(job, s, m)]][0] - sol[waitBefore[(job, s, m)]][0] != 0 :
                        #    print (sol[taskVar[(job, s, m)]][0] , sol[waitBefore[(job, s, m)]][0])
                        #    print (sol[taskVar[(job, s, m)]][0] - sol[waitBefore[(job, s, m)]][0])

                        # print(
                        #    f"{job} {sol[waitBefore[(job, s, m)]][0]} - {sol[waitBefore[(job, s, m)]][1]}  {sol[taskVar[(job, s, m)]][0]} - {sol[taskVar[(job, s, m)]][1]}  {sol[waitAfter[(job, s, m)]][0]} - {sol[waitAfter[(job, s, m)]][1]}")
    # for j in job_list:
    #    for s in range(len(job_stage_elig[j]) - 1):
    #        for m in range(len(job_cell[j]) - 1):
    #           if job_cell[j][m] in stages[job_stage_elig[j][s]]:
    #               # Each job starts waiting to enter the machine of stage s+1 when it ends waiting to exit the machine of stage s, added by the respective travel time.
    # model.add(model.start_at_end(waitBefore[(j, job_stage_elig[j][s + 1], job_cell[j][m + 1])],
    #                             waitAfter[(j, job_stage_elig[j][s], job_cell[j][m])],
    #                             -transTimes[job_cell[j][m]][job_cell[j][m + 1]]))
    #                print (j,job_stage_elig[j][s],job_cell[j][m] , sol[waitAfter[(j, job_stage_elig[j][s], job_cell[j][m])]], sol[waitBefore[(j, job_stage_elig[j][s + 1], job_cell[j][m + 1])]], sol[taskVar[(j, job_stage_elig[j][s + 1], job_cell[j][m + 1])]])
    # print (C_max)
    return C_max, sols


'''
    nJobs, jobs = random.randint(100, 150), {}
    nStages = random.randint(3, 5)
    for j in range(nJobs):
        jobs[j] = [None for s in range(nStages)]
        for s in range(nStages):
            processingTime = random.randint(100, 600)
            jobs[j][s] = processingTime  # jobs[j][s] : processing time of job 'j' for stage 's'

    # Random machines
    machines, nMachines = {}, 0
    for s in range(nStages):
        machines[s] = []
        for m in range(random.randint(2, 4)):  # For each stage, we construct 2-4 machines.
            machines[s].append({})
            # machines[stage][number of dedicated machines] = {id of machine : {'entryBuffers' : number of buffers in the entry, 'exitBuffers' : number of buffers in the exit}}
            machines[s][len(machines[s]) - 1][nMachines] = {
                'entryBuffers': random.randint(1, 5) * (int(bool(s > 0))) + nJobs * (int(bool(s == 0))),
                'exitBuffers': random.randint(1, 5) * (int(bool(s < nStages - 1))) + nJobs * (
                    int(bool(s == nStages - 1)))}
            nMachines += 1
    travelTimes = [[None for m in range(nMachines)] for n in range(
        nMachines)]  # Random travel times between machines of consecutive stages (i.e., if 'm1' and 'm2' are dedicated to the same stage or to non-conseuctive stages, then travel time is None)
    for s in range(nStages - 1):
        for m in range(nMachines):
            for k1 in machines[s]:
                for k2 in k1.keys():
                    if k2 == m:
                        for n in range(nMachines):
                            for k3 in machines[s + 1]:
                                for k4 in k3.keys():
                                    if k4 == n:
                                        travelTimes[m][n] = random.randint(30, 180)

    # Generation of a solution
    assignedJobs = {}
    for j in range(nJobs):
        assignedJobs[j] = []
        for s in range(nStages):
            selectedMachine = random.choice([k2 for k1 in machines[s] for k2 in k1.keys()])
            assignedJobs[j].append(selectedMachine)  # For each job, we choose a random machine for each stage.
    sequences = {}
    for m in range(nMachines):
        sequences[m] = []
        for j in range(nJobs):
            if m in assignedJobs[j]:
                sequences[m].append(j)  # sequences[machines] : sequence of jobs, being processes in the machine


        # 'Machine' :     'Job'     'Start_of_waiting_before_entry' - 'End_of_waiting_before_entry'     'Start_of_processing' - 'End_of_processing'      'Start_of_waiting_after_exit' - 'End_of_waiting_after_exit'
        print("------------------------------------")
        for j in job_list:
            for s in job_stage_elig[j]:
                print(f"Stage {s}")
                print("------------------------------------")
                for m in stages[s]:
                    if m in job_cell[j]:
                        print(
                            f"{m} :    {j} {sol[waitBefore[(j, s, m)]][0]} - {sol[waitBefore[(j, s, m)]][1]}  {sol[taskVar[(j, s, m)]][0]} - {sol[taskVar[(j, s, m)]][1]}  {sol[waitAfter[(j, s, m)]][0]} - {sol[waitAfter[(j, s, m)]][1]}")
            print("------------------------------------")

    '''

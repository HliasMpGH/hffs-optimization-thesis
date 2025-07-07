import math


'''
Assign the jobs of a given order sequence to machines (cells)
in all eligible stages of each job.
'''
def ffs_assignment(orders_seq, stages, process, job_stage_elig, transTimes, orders, job_list, WIPs, format_orders=False):

    # the (time) load of each machine (completion time of the last job assigned to it)
    loads = {j: 0 for i in stages.keys() for j in stages[i]}

    # the sequence of jobs assigned at each machine
    job_order = {j: [] for i in stages.keys() for j in stages[i]}

    # the list of machines each job gets assigned to (per job)
    job_cell = {j: [] for j in job_list}

    # start/end times of jobs (per eligible stage)
    starting_times, completion_times = {}, {}

    # sum_of_comp = 0
    for ord in orders_seq:

        if format_orders:
            # FORMAT THE ORDER BECAUSE PYGAD WANTS ONLY NUMBERS FOR GENES
            formatted_order = f"Order-{int(ord)}"

            jobs = orders[formatted_order]
        else:
            jobs = orders[ord]
        for i in jobs:
            starting_times[i], completion_times[i] = {}, {}
            previous_cell = 0
            for s in job_stage_elig[i]:
                starting_times[i][s], completion_times[i][s] = 0, 0
                est_time, cell_to_select = math.inf, 0
                for cell in stages[s]:
                    # if the job is assigned in a stage before this, also consider travel times
                    if previous_cell != 0:
                        if est_time > max(loads[previous_cell] + transTimes[previous_cell][
                                cell], loads[cell]):
                            cell_to_select = cell
                            starting_times[i][s] = max(loads[previous_cell] + transTimes[previous_cell][
                                cell], loads[cell])
                            est_time = starting_times[i][s]
                            completion_times[i][s] = starting_times[i][s] + process[s][i]
                    else:
                        # first assignment of job, no travel times exist
                        if est_time > loads[cell]:
                            cell_to_select = cell
                            est_time = loads[cell]
                            starting_times[i][s] = loads[cell]
                            completion_times[i][s] = starting_times[i][s] + process[s][i]
                # update fields with respect to the assignment
                loads[cell_to_select] = completion_times[i][s]
                # sum_of_comp += completion_times[i][s]
                job_order[cell_to_select].append(i)
                job_cell[i].append(cell_to_select)
                previous_cell = cell_to_select
    #for ord in orders_seq:
    #    # print (orders[ord])
    #    jobs = [j for i in orders[ord] for j in orders[ord][i]]
    #    for i in jobs:
    #        print (i)
    #        for s in job_stage_elig[i]:
    #            print (s, starting_times[i][s], completion_times[i][s])

    # set cmax as the greater load of a machine
    stage_max, C_max = max(loads.items(), key=lambda k: k[1])
    return C_max, job_order, loads, starting_times, completion_times, job_cell


'''
def seq_maker(jobs):
    return random.shuffle(jobs)


def job_assignment(cells_per_stage, jobs, stages, job_stage_elig, process):
    sequence = seq_maker(list(range(jobs)))
    sequence = list(range(jobs))

    # load initialization
    num_columns = max(cells_per_stage)
    num_rows = stages
    loads = {}
    # loads = np.zeros((num_rows, num_columns), dtype=int)
    job_order = {}

    for i in range(0, len(cells_per_stage), 1):
        for j in range(0, cells_per_stage[i]):
            loads[i, j] = 0
            job_order[i, j] = []

    # for i, cells in enumerate(cells_per_stage):
    #   print (i,cells)
    #  loads[i, cells:] = -1
    # job_order[i][cells:] = [-1]

    # assignment
    for j in sequence:
        for stage in job_stage_elig[j]:
            temp_loads = []
            for m in range(0, cells_per_stage[stage]):
                temp_loads.append(loads[stage, m])
            min_load = min(temp_loads)

            # finding the first machine where load = min_load
            flag = False
            w = 0
            while not flag:
                if loads[stage, w] == min_load:
                    loads[stage, w] += process[j][stage]
                    job_order[stage, w].append(j)
                    flag = True
                w += 1
    for i in range(stages):
        for j in range(cells_per_stage[i]):
            print('Stage', i, 'Cell', j)
            print(loads[i,j])
            print(job_order[i,j])
    print(job_stage_elig)



    # makespan calculation
    job = {}
    for i in range(0, len(cells_per_stage), 1):
        for j in range(0, cells_per_stage[i]):
            for w in range(jobs):
                job[i, j, w] = 0

    for i in range(stages):
        for cell in range(0, cells_per_stage[i]):
            fl = 0
            temp = 0
            for j in job_order[i, cell]:
                if fl != 0:
                    job[i, cell, j] = process[j, i] + job[i, cell, temp]
                else:
                    job[i, cell, j] = process[j, i]
                    fl = 1
                temp = j
    for i in range(stages):
        for cell in range(0, cells_per_stage[i]):
            for j in job_order[i, cell]:
                if job[i, cell, j] == 0:
                    job.pop((i, cell, j))
    print(job)
    Cmax = max(job)
    print('Cmax', Cmax)
    return Cmax, job_order

# def call(instance):
#    general, stages, process, job_mapping, stage_mapping, cells_per_stage, job_stage_elig, stage_job_elig, transportation_time \
#       = ffs_preprocessing.data_preprocessing_identical(instance)
#  return job_assignment(cells_per_stage, job_mapping, stages, job_stage_elig, process)
'''

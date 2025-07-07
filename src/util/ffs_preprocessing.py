# import math
# import pandas as pd
from datetime import datetime
import numpy as np
import random
import math

'''
general: dict: 'case', 'numStages', 'mType', 'batch', 'jobsToSchedule', 'schedHorizon', 'numOrders', 'numJobs', 'mInterval'
stages: dict: {'Stage_name': [Cell_names]}
process: np.array --> process times of job j to stage s (identical)
job_stage_elig: dict of lists --> key: number of job based on mapping, item: list of stages where processing time > 0
'''


def data_preprocessing_identical(input_data):
    print(datetime.now(), " - Start of Data Preprocessing.")
    # Assign input data to specific dictionaries
    general, stage_data, process, orders_data, transTimes = input_data['general'], input_data[
        'stages'], input_data['processTimes'], input_data['orders'], input_data['transTimes']
    for i in transTimes.keys():
        for j in transTimes[i].keys():
            transTimes[i][j] = math.ceil(transTimes[i][j])

    WIPs = {}
    maintenance = {}
    stages = {}
    cells_per_stage = {}
    for s in stage_data.keys():
        stages[s] = []
        if 'Cells' in stage_data[s].keys():
            cells_per_stage[s] = len(list(stage_data[s]['Cells']))
            for cell in stage_data[s]['Cells'].keys():
                stages[s].append(cell)
                if 'maintenance' in stage_data[s]['Cells'][cell]:
                    maintenance[cell] = stage_data[s]['Cells'][cell]['maintenance']

    for s in stages.keys():
        for m in stages[s]:
            WIPs[m] = {}
            if 'WIP_in' in stage_data[s].keys():
                WIPs[m]['in'] = stage_data[s]['WIP_in']
            else:
                WIPs[m]['in'] = 1000
            if 'WIP_out' in stage_data[s].keys():
                WIPs[m]['out'] = stage_data[s]['WIP_out']
            else:
                WIPs[m]['out'] = 1000
    orders = {}
    jobs = []
    for i in orders_data:
        orders[i] = []

        for key, job in orders_data[i]['jobConnect'].items():
            orders[i].append(job)
            jobs.append(job)

    # Initialize np arrays
    job_stage_elig = {i: [] for i in jobs}
    stage_job_elig = {i: [] for i in stages.keys()}
    for s in stages.keys():
        for job in jobs:
            if process[s][job] > 0:
                stage_job_elig[s].append(job)
                job_stage_elig[job].append(s)

    print(datetime.now(), " - End of Data Preprocessing.")
    return general, stages, process, job_stage_elig, stage_job_elig, transTimes, orders, jobs, WIPs, cells_per_stage


def get_stage_of_cell(stage_data, cell):
    for stage, stage_info in stage_data.items():
        for c in stage_info['Cells'].keys():
            if str(c) == str(cell):
                return stage


def get_stage_of_job_init(stage_data, init_loads, job):
    for cell, load in init_loads.items():
        if load['jobs'] == job:
            return get_stage_of_cell(stage_data, cell)
    return -1

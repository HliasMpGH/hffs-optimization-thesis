import random
import json

if __name__ == "__main__":
    big_M = 500000
    final_values = {}
    stages = [3, 6, 9] # 3, 6, 9
    wip_range = [0]  # interval 0: [1,2] capacity wip, interval 1: [3,4] capacity wip
    # num_orders = [10, 30, 50, 100, 150]
    num_orders = [10, 30, 50]
    trials = 3
    tot_max_machines = 3

    tot_process = {}
    for s in [stages[-1]]:  # max stages
        for o in [num_orders[-1]]:  # max orders
            for j in range(num_orders[-1] * 3):  # max jobs
                tot_process[j] = {}
                for k in range(s):
                    if k == 0:
                        tot_process[j][k] = random.randint(2, 15)
                    else:
                        if random.random() < 0.8:
                            tot_process[j][k] = random.randint(2, 15)
                        else:
                            tot_process[j][k] = 0

    tot_transp = {}
    for stage1 in range(stages[-1] - 1):  # max stages
        # print(stage1_num, s - 1)
        for mach1 in range(tot_max_machines):
            tot_transp[f"{stage1}-{mach1}"] = {}
            for stage2 in range(stages[-1]):
                if stage2 > stage1:
                    for mach2 in range(3):
                        tot_transp[f"{stage1}-{mach1}"][f"{stage2}-{mach2}"] = random.randint(1, 4) * (
                        (stage2 - stage1))

    order_set = {i: {} for i in range(num_orders[-1])}
    counter = 0
    for i in order_set.keys():
        job_num = random.randint(1, 3)
        order_set[i] = list(range(counter, counter + job_num))
        counter += job_num

    total_values = {'orders': {}, 'processTimes': {}, 'transTimes': {}}
    for i in order_set.keys():
        total_values['orders'][f"Order-{i}"] = {'jobConnect': {}}
        counter = 0
        for j in order_set[i]:
            total_values['orders'][f"Order-{i}"]['jobConnect'][counter] = f"Order_{i}-Job_{j}"
            counter += 1

    for s in range(stages[-1]):  # max stages
        total_values['processTimes'][f"Stage_{s}"] = {}
        for i in order_set.keys():
            for j in order_set[i]:
                total_values['processTimes'][f"Stage_{s}"][f"Order_{i}-Job_{j}"] = tot_process[j][s]

    for stage1 in range(stages[-1] - 1):  # max stages
        for mach1 in range(tot_max_machines):
            total_values['transTimes'][f"Stage_{stage1}-Machine_{mach1}"] = {}
            for stage2 in range(stages[-1]):
                if stage2 > stage1:
                    for mach2 in range(3):
                        total_values['transTimes'][f"Stage_{stage1}-Machine_{mach1}"][
                            f"Stage_{stage2}-Machine_{mach2}"] = tot_transp[f"{stage1}-{mach1}"][f"{stage2}-{mach2}"]

    with open(f"input_benchmark_total.json", "w") as outfile:
        json.dump(total_values, outfile, indent=4)
    for trial in range(trials):
        for ordr in num_orders:
            orders = {}
            inst_orders = random.sample(list(order_set.keys()), ordr)
            inst_jobs = []
            for o in inst_orders:
                orders[f"Order-{o}"] = {'jobConnect': {}}
                counter = 0
                for j in order_set[o]:
                    orders[f"Order-{o}"]['jobConnect'][counter] = f"Order_{o}-Job_{j}"
                    counter += 1
                    # job_list.append(f"Order_{o}-Job_{j}")
                    inst_jobs.append(j)

            for s in stages:
                num_machines = {s: 1 for s in range(s)}
                instance_machines = random.randint(2 * s, 3 * s)
                max_machines = instance_machines - s
                while max_machines > 0:
                    st = random.randint(0, s - 1)
                    if num_machines[st] < 3:
                        num_machines[st] += 1
                        max_machines -= 1

                trans_times = {}
                for stage1 in range(s - 1):  # max stages
                    for mach1 in range(num_machines[stage1]):
                        trans_times[f"Stage_{stage1}-Machine_{mach1}"] = {}
                        for stage2 in range(s):
                            if stage2 > stage1:
                                for mach2 in range(num_machines[stage2]):
                                    trans_times[f"Stage_{stage1}-Machine_{mach1}"][f"Stage_{stage2}-Machine_{mach2}"] = \
                                        tot_transp[f"{stage1}-{mach1}"][f"{stage2}-{mach2}"]
                process = {}
                for st_o in range(s):
                    process[f"Stage_{st_o}"] = {}
                    for o in inst_orders:
                        for j in order_set[o]:
                            process[f"Stage_{st_o}"][f"Order_{o}-Job_{j}"] = tot_process[j][st_o]
                for wip in wip_range:
                    instance_number = "S_" + str(s) + "-Ord_" + str(ordr) + "-I_" + str(trial) + "-WIP_" + str(wip)
                    stage_att = {}
                    for stage in range(s):
                        if wip == 0:
                            wip_in = random.randint(1, 2)
                            wip_out = random.randint(1, 2)
                        else:
                            wip_in = random.randint(3, 4)
                            wip_out = random.randint(3, 4)
                        if stage == 0:
                            wip_in = big_M
                        elif stage == s - 1:
                            wip_out = big_M
                        stage_att[f"Stage_{stage}"] = {'WIP_in': wip_in, 'WIP_out': wip_out, 'Cells': {}}
                        for mach in range(num_machines[stage]):
                            stage_att[f"Stage_{stage}"]['Cells'][f"Stage_{stage}-Machine_{mach}"] = {}

                        # final dataset creation
                        final_values[instance_number] = {}
                        final_values[instance_number]['general'] = {'numStages': s, 'numOrders': ordr,
                                                                    'numJobs': len(inst_jobs),
                                                                    'numMachines': instance_machines, 'wip': wip,
                                                                    'trial': trial}
                        final_values[instance_number]['stages'] = stage_att
                        final_values[instance_number]['orders'] = orders
                        final_values[instance_number]['processTimes'] = process
                        final_values[instance_number]['transTimes'] = trans_times
            '''
            safaf


                tr_times = {}
                for stage1 in stage_att.keys():
                    stage1_num = int(stage1.split('_')[1])
                    if stage1_num < s - 1:
                        # print(stage1_num, s - 1)
                        for mach1 in stage_att[stage1]['Cells']:
                            tr_times[mach1] = {}
                            for stage2 in stage_att.keys():
                                stage2_num = int(stage2.split('_')[1])
                                if stage2_num > stage1_num:
                                    for mach2 in stage_att[stage2]['Cells']:

                                        if stage2_num - stage1_num < 2:
                                            tr_times[mach1][mach2] = random.randint(1,10)
                                        elif stage2_num - stage1_num < 4:
                                            tr_times[mach1][mach2] = random.randint(10,20)
                                        elif stage2_num - stage1_num < 6:
                                            tr_times[mach1][mach2] = random.randint(20,30)
                                        else:
                                            tr_times[mach1][mach2] = random.randint(30,40)

                                        tr_times[mach1][mach2] = random.randint(2, 5) * ((stage2_num - stage1_num) ** 2)
                for trial in range(trials):
                    for ordr in num_orders:
                        instance_number = "S_" + str(s) + "-WIP_" + str(wip) + "-Ord_" + str(ordr) + "-I_" + str(trial)
                        # print(instance_number)
                        orders, job_list = {}, []
                        for o in range(ordr):
                            orders[f"Order-{o}"] = {'jobConnect': {}}
                            jobs = random.randint(1, 3)
                            for j in range(jobs):
                                orders[f"Order-{o}"]['jobConnect'][j] = f"Order_{o}-Job_{j}"
                                job_list.append(f"Order_{o}-Job_{j}")
                        process = {}
                        for stage1 in stage_att.keys():
                            stage1_num = int(stage1.split('_')[1])
                            process[stage1] = {}
                            for j in job_list:
                                if stage1_num == 0:
                                    process[stage1][j] = random.randint(2, 15)
                                else:
                                    if random.random() < 0.8:
                                        process[stage1][j] = random.randint(2, 15)
                                    else:
                                        process[stage1][j] = 0
                            with open(f"input_benchmark_HFFS.json", "w") as outfile:
                                json.dump(orders, outfile)
                            '''

    with open(f"thesis_benchmarks.json", "w") as outfile:
        json.dump(final_values, outfile, indent=4)

    print(len(final_values.keys()))

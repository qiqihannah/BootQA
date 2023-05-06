import os

import pandas as pd
import dimod
from dwave.system import DWaveSampler
import dwave.inspector
import hybrid
import argparse

def get_data(data_name):
    data = pd.read_csv("code/dataset/"+data_name+".csv", dtype={"time": float, "rate": float})
    data = data.drop(data[data['rate'] == 0].index)
    return data

def create_bqm(data):
    '''
    :param data: dataframe
    :return: a bqm of the objective function
    '''
    dic_time = {}
    dic_rate = {}
    dic_num = {}
    time_total = 0
    rate_total = 0
    for id in range(len(data)):
        dic_time["T"+str(id)] = data["time"].iloc[id]
        dic_rate["T"+str(id)] = data["rate"].iloc[id]
        time_total += data["time"].iloc[id]
        rate_total += data["rate"].iloc[id]
        dic_num["T"+str(id)] = 1
    bqm_time = dimod.BinaryQuadraticModel(dic_time,{}, 0,dimod.Vartype.BINARY)
    bqm_rate = dimod.BinaryQuadraticModel(dic_rate,{}, 0,dimod.Vartype.BINARY)
    bqm_num = dimod.BinaryQuadraticModel(dic_num,{}, 0,dimod.Vartype.BINARY)

    bqm_time.normalize()
    bqm_time.normalize()
    bqm = (1 / 3) * pow((bqm_time - 0) / len(data), 2) + (1 / 3) * pow((bqm_rate - rate_total) / len(data), 2) + (1 / 3) * pow((bqm_num - 0) / len(data), 2)

    return bqm

def info_sample(si):
    print(si)
    print(si.samples.first.energy)
    print(si.samples.info)

    dwave.inspector.show(si)
    return si.samples.first.energy

def run_qpu(data, sample_size, repeat):
    '''
    :param sample_list_total: all sampled test cases
    :param data: dataframe
    :return: energy and sample of the best solution
    '''
    obj = create_bqm(data)

    max_subproblem_size = sample_size
    qpu_sampler = DWaveSampler(token='XXX')
    qpu_reads = 100
    qpu_params = None
    max_iter = 100
    max_time = None
    convergence = 3
    energy_reached = None

    initial_state = hybrid.State.from_sample(hybrid.random_sample(obj), obj)
    initial_selected = [int(x) for x in [list(initial_state.samples.first.sample.keys())[id][1:] for id in range(len(data)) if
                      list(initial_state.samples.first.sample.values())[id] == 1]]

    qpu_info_tracker = hybrid.Lambda(lambda _, s: s.updated(
        num_qubits=s.num_qubits + [s['subsamples'].info['embedding_context']['embedding']],
        sampling_time = s.sampling_time + [s['subsamples'].info['timing']['qpu_access_time']*(1e-6)],
        timing = s.timing + [s['subsamples'].info['timing']],
        problem_id = s.problem_id + [s['subsamples'].info['problem_id']],
        embedding_context = s.embedding_context + [s['subsamples'].info['embedding_context']],
        warnings = s.warnings + [s['subsamples'].info['warnings']],
        sampleset_info = s.sampleset_info + [s['subsamples']],
    ))

    result_tracker = hybrid.Lambda(lambda _, s: s.updated(
        first_sample=s.first_sample + [s['samples'].first.sample],
        first_energy=s.first_energy + [s['samples'].first.energy]
    ))

    decomposer = hybrid.EnergyImpactDecomposer(size=max_subproblem_size, rolling=True, rolling_history=0.3, traversal='bfs')
    sampler = hybrid.QPUSubproblemAutoEmbeddingSampler(
                num_reads=qpu_reads, qpu_sampler=qpu_sampler, sampling_params=qpu_params)
    composer = hybrid.SplatComposer()
    iteration = (decomposer
                 | sampler
                 | qpu_info_tracker | composer|result_tracker)

    workflow = hybrid.Loop(iteration, max_iter=max_iter, max_time=max_time,
                           convergence=convergence, terminate=energy_reached)

    state_updated = workflow.run(initial_state.updated(
        num_qubits=[], sampling_time = [], timing = [], problem_id = [], embedding_context = [], warnings = [], sampleset_info = [], first_sample=[], first_energy=[])).result()


    num_qubit_list = state_updated.num_qubits
    sampling_time_list = state_updated.sampling_time
    timing_list = state_updated.timing
    problem_id_list = state_updated.problem_id
    embedding_context_list = state_updated.embedding_context
    warnings_list = state_updated.warnings
    sampleset_info_list = state_updated.sampleset_info
    first_sample_list = state_updated.first_sample
    first_energy_list = state_updated.first_energy
    qubit_sum = []
    subproblem_variables = []

    for variable in num_qubit_list:
        qubit_list = variable.values()
        subproblem_variables.append([int(case[1:]) for case in variable.keys()])
        sum_q = 0
        for q in qubit_list:
            sum_q += len(q)
        qubit_sum.append(sum_q)

    df_log = pd.DataFrame()

    if not os.path.exists("../EIDQ/" + data_name + "/size_" + str(sample_size) + "/" + str(repeat) + "/sample_info/"):
        os.makedirs("../EIDQ//" + data_name + "/size_" + str(sample_size) + "/" + str(repeat) + "/sample_info/")

    selected_list = []
    fval_list = []
    selected_num = []
    selected_time = []
    selected_rate = []
    total_selected_time = []
    total_selected_cases = []
    total_selected_rate = []
    total_fval_list = first_energy_list
    print("state updated and energy")
    print(state_updated)
    print(first_energy_list)
    for subsampleset_index in range(len(sampleset_info_list)):
        total_selected_cases_item = [int(x) for x in [list(first_sample_list[subsampleset_index].keys())[id][1:] for id in range(len(data)) if list(first_sample_list[subsampleset_index].values())[id] == 1]]
        total_selected_time_item = [data["time"].iloc[total_selected_cases_item[index]] for index in range(len(total_selected_cases_item))]
        total_selected_rate_item = [data["rate"].iloc[total_selected_cases_item[index]] for index in range(len(total_selected_cases_item))]
        total_selected_cases.append(total_selected_cases_item)
        total_selected_time.append(total_selected_time_item)
        total_selected_rate.append(total_selected_rate_item)
        subsampleset = sampleset_info_list[subsampleset_index]
        selected_list_item = [int(x) for x in [list(subsampleset.first.sample.keys())[id][1:] for id in range(sample_size) if list(subsampleset.first.sample.values())[id] == 1]]
        selected_list.append(selected_list_item)
        selected_num.append(len(selected_list_item))
        fval_list.append(subsampleset.first.energy)
        subsampleset.to_pandas_dataframe().to_csv('../EIDQ/' + data_name + '/size_' + str(sample_size) + "/" + str(repeat) + "/sample_info/" + "loop_" + str(subsampleset_index) + ".csv")
        selected_time_item = [data["time"].iloc[selected_list_item[index]] for index in range(len(selected_list_item))]
        selected_rate_item = [data["rate"].iloc[selected_list_item[index]] for index in range(len(selected_list_item))]
        selected_time.append(selected_time_item)
        selected_rate.append(selected_rate_item)


    decomposer_time = decomposer.timers["dispatch.next"]
    sampler_time = sampler.timers["dispatch.next"]
    composer_time = composer.timers["dispatch.next"]
    qpu_info_tracker_time = qpu_info_tracker.timers["dispatch.next"]
    result_tracker_time = result_tracker.timers["dispatch.next"]
    iteration_total_time = iteration.timers["dispatch.next"]

    df_log["selected_cases"] = total_selected_cases
    df_log["selected_time"] = total_selected_time
    df_log["selected_rate"] = total_selected_rate
    df_log["fval"] = total_fval_list
    df_log["subproblem_list"] = subproblem_variables
    df_log["sub_selected_list"] = selected_list
    df_log["sub_selected_num"] = selected_num
    df_log["sub_fval"] = fval_list
    df_log["qubit_num"] = qubit_sum
    df_log["spent_time"] = [iteration_total_time[i]-qpu_info_tracker_time[i]-result_tracker_time[i] for i in range(len(iteration_total_time))]
    df_log["iteration_time"] = iteration_total_time
    df_log["decomposer_time"] = decomposer_time
    df_log["sampler_spent_time"] = sampler_time
    df_log["sampling_time"] = sampling_time_list
    df_log["embedding_time"] = [sampler_time[i] - sampling_time_list[i] for i in range(len(sampler_time))]
    df_log["composer_time"] = composer_time
    df_log["qpu_info_tracker_time"] = qpu_info_tracker_time
    df_log["result_tracker_time"] = result_tracker_time
    df_log["qpu_timing"] = timing_list
    df_log["problem_id"] = problem_id_list
    df_log["embedding_context"] = embedding_context_list
    df_log["warnings"] = warnings_list


    df_log.to_csv("../EIDQ/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat)+"/log.csv")

    energy_first = state_updated.samples.first.energy
    sample_first = state_updated.samples.first.sample
    execution_time = workflow.timers["dispatch.next"]

    return initial_selected, sample_first, energy_first, execution_time



def gen_dic(data):
    foods = {}
    for i,x in enumerate(data[["time","rate"]].to_dict(orient="records")):
        foods["T{}".format(i)] = x
    return foods


def print_diet(sample,data, data_name, sample_size, repeat, execution_time, initial_selected):
    result_df = pd.DataFrame(columns=["index", "time", "rate"])
    sum_df = pd.DataFrame(columns=["selected_case_num", "total_time", "total_rate", "fval", "execution_spent_time", "initial_selected"])
    foods = gen_dic(data)
    total_time = 0
    total_rate = 0
    time_list = []
    rate_list = []
    count = 0
    for t in sample.keys():
        if t[0] == 'T' and sample[t] == 1:
            total_time += foods[t]['time']
            total_rate += foods[t]['rate']
            time_item = foods[t]['time']
            rate_item = foods[t]['rate']
            time_list.append(time_item)
            rate_list.append(rate_item)
            count += 1
            result_df.loc[len(result_df)] = [t[1:], time_item, rate_item]

    time_list_n = [time_list[index]/max(data["time"]) for index in range(len(time_list))]
    fval = (1/3)*pow(sum(time_list_n)/len(data),2) + (1/3)*pow((sum(rate_list)-sum(data["rate"]))/len(data),2)+(1/3)*pow(count/len(data),2)
    print("fval: "+str(fval))
    sum_df.loc[len(sum_df)] = [count, total_time, total_rate, fval, execution_time[0], initial_selected]
    result_df.to_csv("../EIDQ/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat)+"/result.csv")
    sum_df.to_csv("../EIDQ/" + data_name + "/size_" + str(sample_size) + "/" + str(repeat) + "/sum.csv")

def merge(sample_list):
    case_list = {}
    for i in range(len(sample_list)):
        for t in sample_list[i].keys():
            if t[0] == 'T' and sample_list[i][t] == 1:
                case_list[t] = 1
    return case_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('s', type=int)
    parser.add_argument('r', type=int)
    parser.add_argument('dn', type=str)
    args = parser.parse_args()
    sample_size = args.s
    repeat = args.r
    data_name = args.dn

    data = get_data(data_name)
    initial_selected, sample_first, energy_first, execution_time = run_qpu(data, sample_size, repeat)
    print("energy first", energy_first)
    print_diet(sample_first, data, data_name, sample_size, repeat, execution_time, initial_selected)



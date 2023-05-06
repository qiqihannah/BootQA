import os

import pandas as pd
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import random
import matplotlib.pyplot as plt
import math
import time
from dimod.serialization.format import Formatter
import argparse

def get_data(data_name):
    data = pd.read_csv("code/dataset/"+data_name+".csv", dtype={"time": float, "rate": float})
    data = data.drop(data[data['rate'] == 0].index)
    return data

def bootstrap_sampling(data, sample_time, sample_size):
    '''
    param data: data frame
    :param sample_time: number of sampling
    :param sample_size: number of test cases in a sample
    :return: a two-dimension list of all sampled test cases
    '''
    sample_list_total = []
    for i in range(sample_time):
        sample_list=random.sample(range(0,len(data["time"])),sample_size)
        sample_list_total.append(sample_list)
    return sample_list_total


def create_bqm(sample, sample_size,sample_time, data):
    '''
    :param sample: a list of sampled test cases
    :param sample_time:
    :param data: dataframe
    :return: a bqm of the objective function
    '''
    dic_time = {}
    dic_rate = {}
    dic_num = {}
    time_total = 0
    rate_total = 0
    for id in sample:
        dic_time["T"+str(id)] = data["time"].iloc[id]
        time_total += data["time"].iloc[id]
        dic_rate["T"+str(id)] = data["rate"].iloc[id]
        rate_total += data["rate"].iloc[id]
        dic_num["T"+str(id)] = 1

    bqm_time = dimod.BinaryQuadraticModel(dic_time,{}, 0,dimod.Vartype.BINARY)
    bqm_rate = dimod.BinaryQuadraticModel(dic_rate,{}, 0,dimod.Vartype.BINARY)
    bqm_num = dimod.BinaryQuadraticModel(dic_num,{}, 0,dimod.Vartype.BINARY)

    bqm_time.normalize()
    bqm = (1/3)*pow((bqm_time-0)/sample_size, 2) + (1/3)*pow((bqm_rate - rate_total)/sample_size, 2) + (1/3)*pow((bqm_num - 0)/sample_size, 2)

    return bqm


def run_qpu(sample_list_total, data, sample_time, sample_size, data_name):
    '''
    :param sample_list_total: all sampled test cases
    :param data: dataframe
    :return: energy and sample of the best solution
    '''
    sample_first_list = []
    energy_first_list = []
    execution_time = 0
    execution_time_list = []
    for i in range(len(sample_list_total)):
        obj = create_bqm(sample_list_total[i], sample_size,sample_time, data)
        start = time.time()
        sampler = EmbeddingComposite(DWaveSampler(token="XXX"))
        embedding = time.time()
        sampleset = sampler.sample(obj, num_reads=100)
        Formatter(sorted_by=None).fprint(sampleset)
        end = time.time()
        sampling_time = end - embedding
        spent_time = end - start
        qpu_access = sampleset.info['timing']['qpu_access_time']
        embedding_time = spent_time - qpu_access
        execution_time += spent_time
        execution_time_list.append(spent_time)

        first_sample = sampleset.first.sample
        first_energy = sampleset.first.energy
        sample_first_list.append(first_sample)
        energy_first_list.append(first_energy)

        sample_list = sample_list_total[i]
        selected_list = [int(x) for x in [list(first_sample.keys())[id][1:] for id in range(sample_size) if list(first_sample.values())[id] == 1]]
        selected_num = len(selected_list)
        selected_time = [data["time"].iloc[selected_list[index]] for index in range(selected_num)]
        selected_rate = [data["rate"].iloc[selected_list[index]] for index in range(selected_num)]
        fval = first_energy
        embedding = sampleset.info['embedding_context']['embedding']

        qubit_num = sum(len(chain) for chain in embedding.values())

        if i == 0:
            head_df = ["sample_list", "selected_list", "selected_num", "selected_time", "selected_rate", "fval", "qubit_num", "spent_time(s)", "embedding_time(s)", "sampling_time(s)"]
            head_df += list(sampleset.info.keys())
            df_log = pd.DataFrame(columns=head_df)
        values_df = [sample_list, selected_list, selected_num, selected_time, selected_rate, fval, qubit_num, spent_time, embedding_time, sampling_time]
        values_df += list(sampleset.info.values())
        df_log.loc[len(df_log)] = values_df
        if not os.path.exists("../BootQA/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat)):
            os.makedirs("../BootQA/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat))
        df_log.to_csv("../BootQA/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat)+"/log.csv")

        if not os.path.exists("../BootQA/" + data_name + "/size_" + str(sample_size) + "/" + str(repeat)+"/sample_info/"):
            os.mkdir("../BootQA/" + data_name + "/size_" + str(sample_size) + "/" + str(repeat)+"/sample_info/")
        sampleset.to_pandas_dataframe().to_csv('../BootQA/' + data_name + '/size_' + str(sample_size) + "/" + str(repeat) + "/sample_info/"+"sample_"+str(i)+".csv")

        print(f"Number of logical variables: {len(embedding.keys())}")
        print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")

    return sample_first_list, energy_first_list, execution_time, max(execution_time_list)

def gen_dic(data):
    foods = {}
    for i,x in enumerate(data[["time","rate"]].to_dict(orient="records")):
        foods["T{}".format(i)] = x
    return foods

def print_diet(sample,data, data_name, sample_size, repeat, execution_time, max_time):
    result_df = pd.DataFrame(columns=["index", "time", "rate"])
    sum_df = pd.DataFrame(columns=["selected_case_num", "total_time", "total_rate", "fval", "execution_time", "max_time"])
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
    sum_df.loc[len(sum_df)] = [count, total_time, total_rate, fval, execution_time, max_time]
    result_df.to_csv("../BootQA/"+data_name+"/size_"+str(sample_size)+"/"+str(repeat)+"/result.csv")
    sum_df.to_csv("../BootQA/" + data_name + "/size_" + str(sample_size) + "/" + str(repeat) + "/sum.csv")

def merge(sample_list):
    case_list = {}
    for i in range(len(sample_list)):
        for t in sample_list[i].keys():
            if t[0] == 'T' and sample_list[i][t] == 1:
                case_list[t] = 1
    return case_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=int)
    parser.add_argument('r', type=int)
    parser.add_argument('dn', type=str)
    args = parser.parse_args()
    index = args.i
    index = int(index/10)
    repeat = args.r
    data_name = args.dn
    if data_name == "gsdtsr":
        sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        sample_time_list = [64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 3]
    elif data_name == "iofrol":
        sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        sample_time_list = [383, 191, 127, 95, 76, 63, 54, 47, 42, 38, 34, 31, 29, 27, 25, 23]
    elif data_name == "paintcontrol":
        sample_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 89]
        sample_time_list = [20, 9, 6, 4, 3, 3, 2, 2, 1]
    sample_time = sample_time_list[index]
    sample_size = sample_size_list[index]
    print("repeat times:"+str(repeat))
    print("sample size",sample_size)
    print("data name",data_name)
    data = get_data(data_name)
    sample_total_list = bootstrap_sampling(data, sample_time, sample_size)
    sample_first_list, energy_first_list, execution_time, max_time = run_qpu(sample_total_list, data, sample_time, sample_size, data_name)
    merge_sample = merge(sample_first_list)
    print_diet(merge_sample, data, data_name, sample_size, repeat, execution_time, max_time)


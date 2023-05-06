import time
import pandas as pd
import dimod
import argparse
import neal
import os
def get_data(file_name):
    data = pd.read_csv("code/dataset/"+file_name+".csv", dtype={"time": float, "rate": float})
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
    bqm = (1/3)*pow((bqm_time - 0)/len(data), 2) + (1/3)*pow((bqm_rate - rate_total)/len(data), 2) + (1/3)*pow((bqm_num - 0)/len(data), 2)
    return bqm

def run_qpu(data, repeat):
    '''
    :param sample_list_total: all sampled test cases
    :param data: dataframe
    :return: energy and sample of the best solution
    '''

    obj = create_bqm(data)
    init_state = {}
    for i in obj.variables:
        init_state[i] = 1

    sampler_s = neal.SimulatedAnnealingSampler()

    start = time.time()
    sampleset_s = sampler_s.sample(obj, num_reads=100, initial_states_generator="random")
    end = time.time()
    sample_first = sampleset_s.first.sample
    energy_first = sampleset_s.first.energy

    count = 0
    for t in range(len(sample_first)):
        count += 1

    df = pd.DataFrame()
    df['objective value'] = energy_first
    df['selected_test_case_num'] = count
    df['extime'] = end - start

    if not os.path.exists("../SA/" + data_name + "/" + str(repeat)):
        os.makedirs("../SA/" + data_name + "/" + str(repeat))
    df.to_csv("../SA/" + data_name + "/" + str(repeat) + "/summary.csv")

    return sample_first, energy_first


def gen_dic(data):
    foods = {}
    for i,x in enumerate(data[["time","rate"]].to_dict(orient="records")):
        foods["T{}".format(i)] = x
    return foods

def print_diet(sample,data,file_name):
    result = open("../SA/"+file_name+"/"+str(repeat)+"/result.csv","w")
    foods = gen_dic(data)
    i = 1
    total_time = 0
    total_rate = 0
    case_list = []
    count = 0
    for t in sample.keys():
        if t[0] == 'T' and sample[t] == 1:
            total_time += foods[t]['time']
            total_rate += foods[t]['rate']
            case_list.append(int(t[1:]))
            print(t[1:]+'. ',end=' ')
            print('time: '+str(foods[t]['time']), end=', ')
            print('rate: '+str(foods[t]['rate']), end='\n')
            result.write(t[1:]+'. '+'time: '+str(foods[t]['time'])+','+'rate: '+str(foods[t]['rate'])+'\n')
            i += 1
            count += 1
    print(count)
    print("Total time: " + str(total_time))
    print("Total rate: " + str(total_rate))
    result.write("Total time: " + str(total_time))
    result.write("Total rate: " + str(total_rate))
    print(case_list)

def merge(sample_list):
    case_list = {}
    for i in range(len(sample_list)):
        for t in sample_list[i].keys():
            if t[0] == 'T' and sample_list[i][t] == 1:
                case_list[t] = 1
    return case_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('r', type=int)
    parser.add_argument('dn', type=str)
    args = parser.parse_args()
    repeat = args.r
    data_name = args.dn
    data = get_data(data_name)
    sample_first, energy_first = run_qpu(data, repeat)
    print("energy:"+str(energy_first))
    print_diet(sample_first, data, data_name)

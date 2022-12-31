import pickle
import os
from constant import datasets
from sklearn.model_selection import train_test_split
test_ratio = 0.2

for dataset in datasets:
    data = pickle.load(open(f'./graph/{dataset}_graph.pkl','rb'))
    
    eRDF_num = {} # the num of rdfs for each entity e.g. [apple: 23] means 23 rdfs contain entity apple
    rRDF_num = {} # the num of rdfs for each relation 
    for rdf in data['RDFs']:
        if rdf[0] not in eRDF_num:
            eRDF_num[rdf[0]] = 1
        else:
            eRDF_num[rdf[0]] += 1

        if rdf[1] not in eRDF_num:
            eRDF_num[rdf[1]] = 1
        else:
            eRDF_num[rdf[1]] += 1

        if rdf[2] not in rRDF_num:
            rRDF_num[rdf[2]] = 1
        else:
            rRDF_num[rdf[2]] += 1
            
    stay = [] # the sample must be put in trainset
    out = [] # the sample could be put in testset
    for rdf in data['RDFs']:
        if eRDF_num[rdf[0]] > 1 and eRDF_num[rdf[1]] > 1 and rRDF_num[rdf[2]] > 1:
            out.append(rdf)
            eRDF_num[rdf[0]] -= 1
            eRDF_num[rdf[1]] -= 1
            rRDF_num[rdf[2]] -= 1
            continue
        stay.append(rdf)   
    #print(len(out),len(stay))
    test_size = len(data['RDFs']) * test_ratio / len(out)
    
    trainset, testset = train_test_split(stay, test_size=test_size)
    trainset += stay
    
    path = f'./openke/benchmarks/{dataset}'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'./openke/benchmarks/{dataset}/train2id.txt','w', encoding='utf-8') as f:
        f.writelines(f'{len(trainset)}\n')
        for rdf in trainset:
            f.writelines(f'{rdf[0]} {rdf[1]} {rdf[2]}\n')

    with open(f'./openke/benchmarks/{dataset}/test2id.txt','w', encoding='utf-8') as f:
        f.writelines(f'{len(testset)}\n')
        for rdf in testset:
            f.writelines(f'{rdf[0]} {rdf[1]} {rdf[2]}\n')

    with open(f'./openke/benchmarks/{dataset}/valid2id.txt','w', encoding='utf-8') as f:
        f.writelines('1\n0 0 0\n')

    with open(f'./openke/benchmarks/{dataset}/relation2id.txt','w', encoding='utf-8') as f:
        f.writelines(f"{len(data['r2GraphID'])}\n")
        for key, value in data['r2GraphID'].items():
            f.writelines(f'{key}\t{value}\n')

    with open(f'./openke/benchmarks/{dataset}/entity2id.txt','w', encoding='utf-8') as f:
        f.writelines(f"{len(data['e2GraphID'])}\n")
        for key, value in data['e2GraphID'].items():
            f.writelines(f'{key}\t{value}\n')        
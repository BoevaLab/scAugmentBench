import os
from glob import glob
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

experiment = 'augmentation-ablation-vol1'
data = ['PBMC-Multiome', 'Neurips-Multiome']
models_list = ['SimCLR', 'MoCo', 'SimSiam', 'NNCLR', 'BYOL', 'VICReg', 'BarlowTwins']
pipeline = 'base'

combine = ['add', 'concat', 'clip']
projection = ['True', 'False']

def compare_model_architectures(projection, combine, data):
    metrics = ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']

    cols = []
    for c in combine:
        for p in projection:
            for d in data:
                for metric in metrics:
                    cols.append(f"{d}_{c}_{p}_{metric}")
                    
    df_list = [pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols)]
    for model in models_list:
        for c in combine:
            for p in projection:
                for d in data:
                    PATH = experiment + '/' + d + '/' + model + '/' + pipeline + '/' + f"integrate_{c}_RNA_False_projection_{p}/"
                    result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.csv'))]
                    
                    if len(result_paths) != 5:
                        print("ERROR")
                        exit()

                    for i, seed_path in enumerate(result_paths):
                        with open(seed_path, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                row.pop('Bio conservation')
                                row.pop('Batch correction')
                                row.pop('Total')

                                for k, v in row.items():
                                    df_list[i].loc[model, f'{d}_{c}_{p}_{k}'] = float(v)

    df_final = []
    for df in df_list:
        df_final.append(scale_result(df, projection, combine, data))

    averages = pd.concat([each.stack() for each in df_final],axis=1)\
             .apply(lambda x:x.mean(),axis=1)\
             .unstack()
    stds = pd.concat([each.stack() for each in df_final],axis=1)\
             .apply(lambda x:x.std(),axis=1)\
             .unstack()
    
    remove_cols = list(filter(lambda x: ("Bio conservation" not in x) and ("Batch correction" not in x) and ("Total" not in x), list(averages.columns)))
    averages = averages.drop(remove_cols, axis=1).round(3)
    stds = stds.drop(remove_cols, axis=1).round(3)
    return averages, stds

def scale_result(df, projection, combine, data):
    scaled= pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
    for c in combine:
        for p in projection:
            for d in data:
                biometrics = [f"{d}_{c}_{p}_{i}" for i in ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI']]
                batchmetrics = [f"{d}_{c}_{p}_{i}" for i in ['Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']]
                scaled[f"{d}_{c}_{p}_Batch correction"] = scaled[batchmetrics].mean(1)
                scaled[f"{d}_{c}_{p}_Bio conservation"] = scaled[biometrics].mean(1)
                scaled[f"{d}_{c}_{p}_Total"] = 0.6 * scaled[f"{d}_{c}_{p}_Bio conservation"] + 0.4 * scaled[f"{d}_{c}_{p}_Batch correction"]
                df[f"{d}_{c}_{p}_Bio conservation"] = scaled[f"{d}_{c}_{p}_Bio conservation"].copy()
                df[f"{d}_{c}_{p}_Batch correction"] = scaled[f"{d}_{c}_{p}_Batch correction"].copy()
                df[f"{d}_{c}_{p}_Total"] = scaled[f"{d}_{c}_{p}_Total"].copy()
    return df

plot = 2
if plot == 1:   
    # Compare CLIP, Add, Concat for backbone  
    averages, stds = compare_model_architectures(projection = ['False'], combine=combine, data=['Neurips-Multiome'])
    dir_path = "results/bc/avg_compare_integration_projection_False/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    averages.to_csv(f"{dir_path}avg.csv")
    stds.to_csv(f"{dir_path}std.csv")

elif plot == 2:
    # Compare Concat for backbone and projection
    averages, stds = compare_model_architectures(projection = ['False', 'True'], combine=['add'], data=data)
    dir_path = "results/bc/avg_compare_backbone_projection/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    averages.to_csv(f"{dir_path}avg.csv")
    stds.to_csv(f"{dir_path}std.csv")

# result2 = compare_integrations(projection = ['True'])

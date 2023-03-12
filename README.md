# Commonsense Knowledge Graph-based Adapter for Aspect-level Sentiment Classification

**Authors**: Guojun Lu, Haibo Yu, Zehao Yan, and Yun Xue

This repository contains code, models, and description for our paper ["Commonsense Knowledge Graph-based Adapter for Aspect-level Sentiment Classification"](https://doi.org/10.1016/j.neucom.2023.03.002).

If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!
```
@article{LU202367,
title = {Commonsense knowledge graph-based adapter for aspect-level sentiment classification},
journal = {Neurocomputing},
volume = {534},
pages = {67-76},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.03.002},
author = {Guojun Lu and Haibo Yu and Zehao Yan and Yun Xue},
```



The code is still in the process of sorting out. You can open an issue if you have any questions.

#### 

#### Requirements

+ Python 3.6
+ PyTorch 1.10.0



**❗ We have pre-processed the sub-DBpedias, Abstract Embedding, and Knowledge Graph Embedding for three benchmark datasets. If you do not need to reconstruct them, you can skip the Entity Linking and Knowledge Graph Embedding sections. The preprocessed file can be downloaded [here](https://drive.google.com/file/d/1dM_ishv7-UIn8i1UDiudZYyxG0eq3n6Z/view?usp=sharing), which needs to be unzipped to the ```./graph```**



## 🧰 Entity Linking 

We use the Entity Linking tool [BLINK](https://github.com/facebookresearch/BLINK) to map the **aspect** to **WikiID**, and download the neighbors by **WikiID** from **DBpedia**. The example code is placed in ```./blink_example```.



## 📁 Knowledge Graph Embedding

We use the open-source framework [OpenKE](https://github.com/thunlp/OpenKE) to pre-trained the knowledge graph embedding for sub-DBpedia. The pre-processed files and example code are place in ```./ckga_openke_files```.



## ❤️ CKGA

The **C**ommonsense **K**nowledge **G**raph **A**dapter (CKGA) can be easily adapted to existing ALSC (or ABSA) models without modifying original models. 

**CKGA** is construed with ```adapter_models.py``` and ```adapter_utils.py```, containing **CONTROLER** and **ADAPTER**.

The Sub-DBpedia collected from DBpedia is located in ```'./graph/'```

**[dataset]_graph.pkl** contains ['RDFs', 'e2GraphID', 'GraphID2e', 'r2GraphID', 'GraphID2r', 'e2wikiID', 'wikiID2e', 'wikiID2GraphID', 'GraphID2wikiID', 'e2abs'].

**[dataset]_abs_bert.pkl** contains embeddings of abstract using pre-trained BERT-base.

**[dataset]\_kge\_[kge_model].pkl** contains embeddings of entities using OpenKE.



We have added CKGA to three open-source models [RGAT](https://github.com/shenwzh3/RGAT-ABSA), [BiGCN](https://github.com/NLPWM-WHU/BiGCN), [AFGCN and InterGCN](https://github.com/BinLiang-NLP/InterGCN-ABSA). Due to the upload  limitation, we only upload the modified code. You can find them in the directory ```./original_models```. Replace the original repository code with running  ```run_ckga.sh``` and the original models can be run with CKGA.

The part of adding CKGA is noted in the following code:

```python3
###################### changed
modified part of the code 
######################
```



❗ We also added a more concise and objective example ```./ckga_example.ipynb``` to illustrate how to **adapt CKGA to BERT**.





#### **Explanation of hyper-parameters**

| hyper-param         | Explanation                                                  |
| ------------------- | ------------------------------------------------------------ |
| train_model         | "j" for Joint-training, "d" for independent training and fine-tuning. |
| origin_model_path   | if train_model="d", the origin model checkpoint path should set in this param. |
| origin_model_lr     | if train_model="d", origin_model_lr decides the learning rate of original model, and lr decides the learning rate of CKGA. |
| adapter_mode        | select "origin" to only use the original model, select "adapter" to add CKGA. |
| adapter_kge         | select one from ['transe', 'transh','transr','rotate'] as KGE. |
| adapter_norm        | choose whether to normalize the embedding.                   |
| adapter_gcn_hid_dim | the hidden dim of GCNs in CKGA                               |
| adapter_score       | the threshold of linking results                             |
| adapter_gcn_out_dim | the last hidden state dim of GCNs in CKGA                    |
| adapter_dropout     | dropout of CKGA                                              |
| adapter_layer_num   | the number of GCN layers in CKGA                             |
| fuse_mode           | choose whether the outputs of CKGA and the original model are summed or spliced."p" for plus and "c" for concatenating. |

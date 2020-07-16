# MRC4ERE++
The repository contains the code for Paper "Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction", Accepted by [IJCAI 2020](https://www.ijcai.org/Proceedings/2020/0546.pdf). <br>

If you find this repo helpful, please cite the following:
```text
@inproceedings{zhao-etal-2020-asking,
    title = "Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction",
    author = "Zhao, Tianyang  and
      Yan, Zhao  and
      Cao, Yunbo  and
      Li, Zhoujun",
    booktitle = "Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence",
    month = jan,
    year = "2021",
    address = "Kyoto, Japan",
    publisher = "International Joint Conferences on Artificial Intelligence",
    url = "https://www.ijcai.org/Proceedings/2020/0546.pdf",
    pages = "3948--3954"
}
```
 

## Overview

In this paper, we improve the existing MRCbased entity-relation extraction model through diverse question answering. First, a diversity question answering mechanism is introduced to detect entity spans and two answering selection strategies are designed to integrate different answers. Then, we propose to predict a subset of potential relations and filter out irrelevant ones to generate questions effectively. Finally, entity and relation extractions are integrated in an end-to-end way and optimized through joint learning.<br> 

![Aaron Swartz](https://github.com/TanyaZhao/MRC4ERE_plus/raw/master/model_framework.png)

For example, when extracting a person entity, we can construct diverse questions as follows:
- Who is mentioned in the context?
- Find people mentioned in the context?
- Which words are person entities?

After extracted the head entities, we generate diverse questions to identify tail entities by querying about protential relations.
For example, given the person ```Paul Vercammen``` and the relation ```Lived_In```, questions can be constructed as:
- Find locations which Paul Vercammen is lived in ?
- Where does Paul Vercammen live ?
- Where is Paul Vercammen's home ?


## Contents
1. [Experimental Results](#experimental-results)
2. [Dependencies](#dependencies)
3. [Usage](#usage)


## Experimental Results

We evaluate the proposed method on two widely-used datasets for entity relation extaction: ACE05 and CoNLL04.
Micro precision, recall and F1-score are used as evaluation metrics. 
  
- Results on **ACE 2005**:

  | *Models* | Enity P | Entity R | Entity F | Relation P | Relation R | Relation F|
  | --- | --- | --- | --- | --- | --- | --- |
  |Sun et al. (2018) |83.9 s|83.2| 83.6| 64.9| 55.1| 59.6|
  |Li et al. (2019) |84.7 |84.9| 84.8 |64.8| 56.2| 60.2 |
  |MRC4ERE++ |85.9 |85.2 |**85.5** |62.0| 62.2| **62.1 (+1.9)**|
  
- Results on **CoNLL 2004**:

  | *Models* | Enity P | Entity R | Entity F | Relation P | Relation R | Relation F|
  | --- | --- | --- | --- | --- | --- | --- |
  |Zhang et al. (2017) |– |–| 85.6 |– |–| 67.8|
  |Li et al. (2019) | 89.0 | 86.6 | 87.8 | 69.2 | 68.2 | 68.9 |
  |MRC4ERE++ |89.3 |88.5|**88.9** |72.2| 71.5| **71.9 (+3.0)**|


## Data Preparation

We take the CoNLL04 dataset as an example:
* We have processed the [original data](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/data/CoNLL04) into the MRC-based formation, as listed in the directory ```datasets/conll04/mrc4ere```.

To use the pretrained language model BERT:
* Download [BERT-Base-Cased, English](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz) pretrained model and unzip it into the directory ```pretrained_bert/bert-base-cased/```. In this way, we can load the BERT from local working directory.
    
## Dependencies 

* Package dependencies: 
```bash 
python >= 3.6
PyTorch == 1.1.0
pytorch-pretrained-bert == 0.6.1 
```


## Usage
As an example, the following command trains the proposed mothod on CoNLL04. 
```bash 
cd run
python run_tagger.py 
```


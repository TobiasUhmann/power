# Table of Contents

- [1. Setup](#1-setup)
  - [1.1. Clone repo](#11-clone-repo)
  - [1.2. Install Python packages](#12-install-python-packages)
  - [1.3. Install Neo4j](#13-install-neo4j)      
- [2. Obtain Data](#2-obtain-data)
- [3. Build and evaluate POWER](#3-train-and-evaluate-power)
  - [3.1. Create split](#31-create-split)
  - [3.2. Build and evaluate ruler](#32-build-and-evaluate-ruler)
    - [3.2.1. Create AnyBURL Dataset](#321-create-anyburl-dataset)
    - [3.2.2. Load graph into Neo4j](#322-load-graph-into-neo4j)
    - [3.2.3. Build ruler](#323-build-ruler)
    - [3.2.4. Evaluate ruler](#324-evaluate-ruler)
  - [3.3. Build and evaluate texter](#33-build-and-evaluate-texter)
    - [3.3.1. Create texter dataset](#331-create-texter-dataset)
    - [3.3.2. Train texter](#332-train-texter)
    - [3.3.3. Evaluate texter against predictable facts](#333-evaluate-texter-against-predictable-facts)
    - [3.3.4. Evaluate texter against all facts](#334-evaluate-texter-against-all-facts)
  - [3.4. Evaluate POWER](#34-evaluate-power)  
- [4. Run the app](#4-run-the-app)

 
# 1. Setup

## 1.1. Clone repo

Clone the repo from GitHub:

```
git clone https://github.com/TobiasUhmann/power.git
```

Change into the directory. All the following commands are exptected to be
run from within the project directory:

```
cd power/
```

## 1.2. Install Python packages

Make sure that you run Python 3.9.  Also, it is recommended to create a 
separate Python environment. For example, create a local Anaconda environment:
  
```
conda create -p conda39/ python=3.9
conda activate conda39/
```

Install the Python packages, e.g. using pip:

```
pip install -r requirements.txt
```

To leverage CUDA support, install PyTorch as described on the official 
website (https://pytorch.org/).

## 1.3. Install Neo4j

Install the Neo4j graph database (https://neo4j.com/download-center/) that
is used to find groundings for the AnyBURL rules.

# 2. Obtain data

Download the IRT dataset (< link >). You can also download the intermediate
and final data (< link >) produced by various experiments. The latter follow
the directory structure that is assumed in the following sections:

```
data/
    anyburl/
        cde/
            rules/                      AnyBURL Rules Dir
            facts.tsv                   AnyBURL Facts TSV             
        ...

    irt/
        split/
            cde/                        IRT Split Dir (CoDEx)
            fb/                         IRT Split Dir (FB15K-237)
        text/
            cde-irt-5-marked/           IRT Text Dir (CoDEx graph, IRT sentences, 5 per entity, marked)
            ...
            
    power/
        ruler/
            cde-test-50/                POWER Ruler Dir (CoDEx graph, test data, 50% known test triples)
                rules/                  POWER Rules Dir
                ruler.pkl               POWER Ruler PKL
            ...
        split/
            cde-50/                     POWER Split Dir (CoDEx graph, 50% known valid/test triples)
            ...
        texter/
            cde-irt-5-marked.pkl        POWER Texter PKL (CoDEx graph, IRT sentences, 5 per entity, marked)
            ...
```

# 3. Train and evaluate POWER

The following sections show how to train POWER on the CoDEx graph with 5
marked sentences per entitiy. CoDEx is chosen over Freebase because
the available relation labels are better readable. Choosing Freebase or
increasing the number of sentences improves performance.

## 3.1. Create Split

Create a POWER Split from an IRT Split:

```
python src/create_split.py \
  data/irt/split/cde/ \
  data/irt/text/cde-irt-5-marked/ \
  data/power/split/cde-50/ \
  --known 50
```

During evaluation, 50% of the validation and test facts will be known,
respectively.

## 3.2. Build and evaluate ruler

First, create dataset for AnyBURL, run it. Resulting rules are then matched
on train and valid facts. For that, those need to be loaded to Neo4j graph
database that can be queried for rule groundings.

### 3.2.1 Create AnyBURL dataset

<create_anyburl_dataset.py>

### 3.2.2. Load graph into Neo4j

<load_neo4j_graph.py>

### 3.2.3. Build ruler

<build_ruler.py>

### 3.2.4. Evaluate ruler

<eval_ruler.py>

## 3.3. Build and evaluate texter

First, create dataset. Specifies some sentences and trueness of most common 
facts for each entity. Then, train texter on it, best will be saved. Then,
evaluate against test created dataset. Finally, also evaluate against all
test facts. Of course, cannot predict 100%.

### 3.3.1. Create texter dataset

<create_texter_dataset.py>

### 3.3.2. Train texter

<train_texter.py>

Hint: during training, validated against predictable facts

### 3.3.3. Evaluate texter against predictable facts

<eval_texter_predictable.py>

### 3.3.4. Evaluate texter against all facts

<eval_texter_all.py>

## 3.4. Evaluate POWER

<eval_power.py>

# 4. Run the app

Run App:
<run_app.py>

Go to localhost:...

Select valid entity and observe predicted facts with given rules and texts
<screenshot>

# Table of Contents

- [1. Setup](#1-setup)
  - [1.1. Clone repo](#11-clone-repo)
  - [1.2. Install Python packages](#12-install-python-packages)
  - [1.3. Install Neo4j graph database](#13-install-neo4j-graph-database)      
- [2. Provide Data](#2-provide-data)
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

## 1.2. Install Python packages

## 1.3. Install Neo4j graph database

# 2. Provide data

IRT dataset: <download link>

additionally: intermediate and final data produced when building POWER:
<download link>

it follows the following directory structure that is assumed in examples:

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

Following sections show how to train POWER on the CoDEx graph with 5
marked sentences per entitiy. CoDEx is chosen over Freebase because
the available relation labels are better readable. Note, that performance
is much better when choosing Freebase or when increasing the number of
sentences per entity.

## 3.1. Create Split

Create a POWER Split from an IRT Split:
<create_split.py>

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

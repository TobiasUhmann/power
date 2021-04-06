# Table of Contents

- [1. Setup](#1-setup)
  - [1.1. Clone repo](#11-clone-repo)
  - [1.2. Install Python packages](#12-install-python-packages)
  - [1.3. Download AnyBURL](#13-download-anyburl)      
  - [1.4. Install Neo4j](#14-install-neo4j)      
- [2. Obtain Data](#2-obtain-data)
- [3. Build and evaluate POWER](#3-train-and-evaluate-power)
  - [3.1. Create split](#31-create-split)
  - [3.2. Build and evaluate ruler](#32-build-and-evaluate-ruler)
    - [3.2.1. Create AnyBURL Dataset](#321-create-anyburl-dataset)
    - [3.2.2. Mine rules](#322-mine-rules)
    - [3.2.3. Load graph into Neo4j](#323-load-graph-into-neo4j)
    - [3.2.4. Prepare ruler](#324-prepare-ruler)
    - [3.2.5. Evaluate ruler](#325-evaluate-ruler)
  - [3.3. Train and evaluate texter](#33-train-and-evaluate-texter)
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

## 1.3. Download AnyBURL

Download AnyBURL (http://web.informatik.uni-mannheim.de/AnyBURL/).

## 1.4. Install Neo4j

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
            cde-test-50.pkl             POWER Ruler PKL (CoDEx graph, test data, 50% known test triples)
            ...
        samples/
            cde-irt-5-marked/           POWER Samples Dir (CoDEx graph, IRT sentences, 5 per entity, marked)
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

```bash
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

### 3.2.1. Create AnyBURL dataset

Create the `AnyBURL Facts TSV` that contains the `POWER Split`'s train fact
in the format expected by AnyBURL.

```bash
python src/create_anyburl_dataset.py \
  data/power/split/cde-50/ \
  data/anyburl/cde/facts.tsv
```

### 3.2.2. Mine rules

Create a `rules/` directory next to the generated `facts.tsv`
and put a `config-learn.properties` with the following 
content into it:

```
PATH_TRAINING  = ../facts.tsv
PATH_OUTPUT    = rules
SNAPSHOTS_AT   = 10,50,100
WORKER_THREADS = 7
```

Run AnyBURL. If you have put the downloaded AnyBURL directory
next to the `power` project directory, the invocation from within
the `rules` directory looks like this:

```bash
java -cp ../../../../../AnyBURL/AnyBURL-RE.jar \
  de.unima.ki.anyburl.LearnReinforced \
  config-learn.properties
```

AnyBURL will search the `facts.tsv` for rules and save the
resulting rules in the `rules/` directory.

### 3.2.3. Load graph into Neo4j

To be able to search for rule groundings, the graph has to
be loaded into the Neo4j graph database.

Create and run a new Neo4j instance that it is
available at `localhost:7687` by the default user `neo4j`
with the password `1234567890`. For evaluation on the
validation data copy `entities.tsv`, `train_facts.tsv` and
`valid_facts_known` from the `POWER Split Directory` to
the database's import directory and run the following command:

```bash
python src/load_neo4j_graph.py \
  bolt://localhost:7687 \
  neo4j \
  1234567890
```

You can subsequently explore the graph using the Cypher query
language in the Cypher shell or in the Neo4j GUI Browser.

### 3.2.4. Prepare ruler

It would be very costly to process all learned rules during
every single entity prediction. Therefore, the rules are
processed once for all entities.

Copy the most comprehensive rules file created by AnyBURL
from the `AnyBURL Rules Directory` to the `POWER Ruler
Directory`:

```bash
cp data/anyburl/cde/rules/rules-100 data/power/ruler/cde-50/rules.tsv
```

Then, make sure that the previously created Neo4j instance is
running and execute the following command:

```bash
python src/prepare_ruler.py \
  data/anyburl/cde/rules/rules-100 \
  bolt://localhost:7687 \
  neo4j \
  1234567890 \
  data/power/split/cde-50/ \
  data/power/ruler/cde-50.pkl
```

### 3.2.5. Evaluate ruler

Evaluate the ruler:

```bash
python src/eval_ruler.py \
  data/power/ruler/cde-50.pkl \
  data/power/split/cde-50/
```

By default, the ruler is evaluated against both, the known and
the unknown evaluation facts. If the known evaluation facts were
included during rule mining, they must not be included in the
evaluation (`--filter-known`)!

The results is the micro F1 score over the ground truth rules.

## 3.3. Train and evaluate `Texter`

The `Texter` is a classifier that is trained on the most common facts in the
knowledge graph. The following sections show how to create the dataset, train
the `Texter` and, finally, evaluate it. As the `Texter` is trained on the
most common classes only, it cannot predict all facts, like the `Ruler`. It
is therefore evaluated twice: Once against the facts it can predict 
(comparable to the validation during training) and once against all facts
(comparable to the evaluation of the `Ruler`).

### 3.3.1. Create texter dataset

```bash
python src/create_texter_dataset.py \
  data/irt/split/cde/ \
  data/irt/text/cde-irt-5-marked/ \
  data/power/samples/cde-irt-5-marked/ \
  --class-count 100 \
  --sent-count 5
```

### 3.3.2. Train texter

Train the texter, e.g. on the 100 most common facts of the CoDEx graph 
with 5 marked sentences per entity.

```bash
python src/train_texter.py \
  data/power/samples/cde-irt-5-marked/ \
  100 \
  5 \
  data/power/split/cde-100/ \
  data/power/texter/cde-irt-5-marked.pkl \
  --epoch-count 2
```

Note: During training, the predicted facts are validated against
the predictable facts.

### 3.3.3. Evaluate texter against predictable facts

<eval_texter_predictable.py>

### 3.3.4. Evaluate texter against all facts

```bash
python src/eval_ruler.py \
  data/power/texter/cde-irt-5-marked.pkl \
  5 \
  data/power/split/cde-0/ \
  data/irt/text/cde-irt-5-marked/
```

## 3.4. Evaluate POWER

```bash
python src/eval_ruler.py \
  data/power/ruler/cde-50.pkl \
  data/power/texter/cde-irt-5-marked.pkl \
  5 \
  data/power/split/cde-50/ \
  data/irt/text/cde-irt-5-marked/
```

# 4. Run the app

Run App:
<run_app.py>

Go to localhost:...

Select valid entity and observe predicted facts with given rules and texts
<screenshot>

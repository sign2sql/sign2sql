# SIGN2SQL: Towards SQL Query Generation from Sign Language

We give the dataset of sign2sql which contains 9383 pairs of sign and sql. We also give the initial rule-based English-ASL(America Sign Language) corpus that we used in constructing the sign2sql dataset: ASLG-PC12.

## Introduction to dataset

* **sign2sql.json** store the JSON format of (GLOSS,SQL) pairs.
* **tables.json** contains the following information for each database.
* **databases**  All table contents are contained in corresponding SQLite3 database files. Can be download from [here](https://drive.google.com/file/d/1FugT0ER4THd4rf5OGCPUAGQuNYjtUpAx/view?usp=sharing).
* **gloss2text.json**  contains 87710 pairs of rule-based English-ASL.

### details

`sign2sql.json` contains the following fields:

* `db_id`: solving the database id of this issue.
* `gloss`: the ASL (America Sign Language) question.
* `gloss_toks`: the ASL tokens.
* `query`: the SQL query corresponding to the question.
* `query_toks`: the SQL query tokens.

Below is an example:
```
{"db_id": "department_management", 
"gloss": "HOW DESC-MANY HEAD DEPARTMENT BE DESC-OLDER THAN 56 ?",
"gloss_toks": ["HOW", "DESC-MANY", "HEAD", "DEPARTMENT",  "BE", "DESC-OLDER", "THAN", "56", "?"], 
"query": "SELECT count(*) FROM head  WHERE age  >  56", 
"query_toks": ["SELECT", "count", "(", "*", ")", "FROM",  "head", "WHERE", "age", ">", "56"], "query_toks_no_value": ["select", "count", "(", "*", ")", "from", "head", "where", "age", ">", "{value}"]
},
```
## Baselines
The code is runnable with Python 3.8, PyTorch 0.10.0. 
And we give the seq2seq baselines contains LSTM+LSTM and LSTM+LSTM+Attn.

#### 1.1 Data Preprocessing
Please run `generate_vocab.py` to construct data vocabulary.
#### 1.2 Run experiment
For example: 
1 training with LSTM+LSTM model
```
pyrhon run.py -eval False -attention_mechanism False -model_dir model
```
2 Testing 
```
pyrhon run.py -eval True -attention_mechanism False -load_model model
```






 


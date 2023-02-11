# SIGN2SQL: A Benchmark for Generating SQL Queries from Sign Language
## Dataset information
- Some samples in our dataset

**Natural Language:**  what's the district with party being democratic  

**Pose video:**  

![image](https://github.com/sign2sql/sign2sql/blob/main/samples/length3_1.gif#pic_left)  

**Query:** SELECT District FROM table_1_1342359_5 WHERE Party = Democratic  

**Natural Language:** what's the rank for iceland  

**Pose video:**  

![image](https://github.com/sign2sql/sign2sql/blob/main/samples/length3_2.gif#pic_left)  

**Query:** SELECT Rank FROM table_1_145439_1 WHERE Country = Iceland  

We release the text2sql dataset in [Google Drive](https://drive.google.com/file/d/1-2-razV07_Sgtl3yW0-kesq0zWxHNktY/view?usp=sharing). The video dataset is so huge(500~600GB) that we haven't find a proper way to release it yet.

## Dataset quality check
100 sign-language pose videos and their corresponding SQL queries are randomly selected for evaluation. The quality of the synthesized pose videos are evaluated in two aspects, including consistency, and fluency, with the 1-5 Likert scale (5 for excellent, 4 for good, 3 for acceptable, 2 for marginal, and 1 for poor). The overall averaged score of our human judgement is 4.3. The results of evaluation of the first ten samples are shown in the table below.
<table>
   <tr>
      <td></td>
      <td colspan="2">Overall</td>
   </tr>
   <tr>
      <td></td>
      <td>Stu1</td>
      <td>Stu 2</td>
   </tr>
   <tr>
      <td>1</td>
      <td>4</td>
      <td>5</td>
   </tr>
   <tr>
      <td>2</td>
      <td>5</td>
      <td>4</td>
   </tr>
   <tr>
      <td>3</td>
      <td>4</td>
      <td>4</td>
   </tr>
   <tr>
      <td>4</td>
      <td>4</td>
      <td>4</td>
   </tr>
   <tr>
      <td>5</td>
      <td>3</td>
      <td>3</td>
   </tr>
   <tr>
      <td>6</td>
      <td>4</td>
      <td>4</td>
   </tr>
   <tr>
      <td>7</td>
      <td>5</td>
      <td>5</td>
   </tr>
   <tr>
      <td>8</td>
      <td>5</td>
      <td>4</td>
   </tr>
   <tr>
      <td>9</td>
      <td>5</td>
      <td>5</td>
   </tr>
   <tr>
      <td>10</td>
      <td>4</td>
      <td>4</td>
   </tr>
</table>

The overall results can be found in the [Google Sheet](https://docs.google.com/spreadsheets/d/17BTVRfzrs7DeePSENT8WXkUb1VyN0NBgGa4L0oP19j8/edit#gid=0). Also, the sampled dataset is released in the [Google Drive](https://drive.google.com/file/d/1xPqij9pz3wZEmy_IVEaYOqlDDaIH--Yg/view?usp=share_link).

## Requirements

- `python3.6` or higher.
- `PyTorch 0.4.0` or higher.
- `CUDA 9.0`
- Python libraries: `babel, matplotlib, defusedxml, tqdm, sqlalchemy==1.2, records==0.5.3 tabulate==0.8.1`
- Example
    - Install [minicoda](https://conda.io/miniconda.html)
    - `conda install pytorch torchvision -c pytorch`
    - `conda install -c conda-forge records==0.5.3`
    - `conda install babel` 
    - `conda install matplotlib`
    - `conda install defusedxml`
    - `conda install tqdm`
- stanford corenlp, please refer to this [repo](https://github.com/stanfordnlp/python-stanford-corenlp)
- For the convenience, we use the PyTorch-converted pre-trained BERT from the [SQLova](https://github.com/naver/sqlova) and the pre-trained BERT parameters are available at [here](https://drive.google.com/file/d/1iJvsf38f16el58H4NPINQ7uzal5-V4v4/view?usp=sharing)

## Data Process
**Note： for executing the code, you may need to modify the input and output path in our code.**
- convert the sign videos(mp4 format) in npy format
    ```shell
    cd sign2sql/dataset
    python wikisql_sign_preprocess.py
    ```

- merge database(.db files) and tables(.tables.jsonl files)
    ```shell
    cd sign2sql/dataset
    python merge_database.py
    ```

- split and annotate dataset
    ```shell
    cd sign2sql/dataset
    python split_and_annotate.py
    ```

## Train
- train the Sign2Text model（Transformer）
    ```shell
    CUDA_VISIBLE_DEVICES=3 nohup python -u train_sign2text.py --do_train --seed 1 --bS 3 --accumulate_gradients 2 --bert_type_abb uS --lr 0.0001 > sign2text.log 2>&1 &
    ```

- train the Text2SQL model（SQLova）
    ```shell
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_text2sql.py --do_train --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb uS --lr 0.001 --lr_bert 0.00001 --max_seq_length 222 > text2sql.log 2>&1 &
    ```

- after finish the above model training，inference with above two models（Transformer+SQLova）
    ```shell
    CUDA_VISIBLE_DEVICES=2 nohup python -u test_sign2text2sql.py --seed 1 --bS 3 --bert_type_abb uS > sign2text2sql.log 2>&1 &
    ```

- train the end-to-end model（Sign2SQLNet）
    ```shell
    CUDA_VISIBLE_DEVICES=0 nohup python -u train_sign2sql.py --do_train --seed 1 --bS 3 --accumulate_gradients 2 --bert_type_abb uS --lr 0.001 --lr_enc 0.0001 --lS_enc 3 --dr_enc 0.0 --model_save_path model/sign2sql > sign2sql.log 2>&1 &
    ```


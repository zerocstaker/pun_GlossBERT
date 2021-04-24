# Pun GlossBERT
Project for Semantic Representation

## Dependencies
- GlossBERT
- Pandas
For generating dataset (mainly heterographic):
- Stanza
- [python-Levenshtein](https://github.com/ztane/python-Levenshtein)
- [metaphone](https://github.com/oubiwann/metaphone)

## Instructions
1. Download the latest GlossBERT checkpoint for the model
2. Download the pun dataset - [SemEval-2017 Task 7 dataset](https://alt.qcri.org/semeval2017/task7/) and compile the scorer
3. Run `transform_dataset.py` on pun dataset to transform SemEval dataset into the usual WSD dataset format ([Raganato et al.](http://lcl.uniroma1.it/wsdeval/)) for both homographic and heteographic. This produces file with `.data.xml` suffix
4. Run `generate_sent_cls_ws.py` to transform the WSD dataset into proper format for GlossBERT for both homographic and heteographic
5. Run `evaluate.sh` to predict and put into proper output format.
6. Run scorer on the predicted output

For convenience, the processed dataset for homographic and heterographic can be found in `processed_dataset`. So step 3 and 4 can be skipped and use that instead

An example script for generating Homographic puns (step 3-6) and evaluation is as follows:

Assuming the directory to model is located in `./model` and pun dataset is located in `./semeval2017_task7/data/test`
```
MODEL_DIR="./model"
DATASET_DIR="./semeval2017_task7/data/test"
OUTPUT_DIR="./output"

# This generate .data.xml
python transform_dataset.py --dataset_dir $DATASET_DIR

# This generate _sent_cls_ws.csv 
python generate_cls_ws.py --dataset_dir $DATASET_DIR

# Run GlossBERT on Homographic, Might need CUDA_VISIBLE_DEVICES= . Defaults to generate ./output
bash evaluate.sh $DATASET_DIR/subtask3-homographic-test_sent_cls_ws.csv

# transform result into what the Scorer expects
python transform_result.py --dataset_file $DATASET_DIR/subtask3-homographic-test_sent_cls_ws.csv --output_dir output

# use Scorer
cd semeval2017_task7/scorer/bin/
java de.tudarmstadt.ukp.semeval2017.task7.scorer.PunScorer -i ../../data/test/subtask3-homographic-test.gold  ../../../output/prediction.txt

###
# coverage: 0.9845916795069337
# precision: 0.1705790297339593
# recall: 0.1679506933744222
# f1: 0.16925465838509318
###

```
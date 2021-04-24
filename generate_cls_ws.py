import os
import argparse
import string
import pandas as pd
from stanza.server import CoreNLPClient
import xml.etree.ElementTree as ET
from collections import defaultdict

from Levenshtein import distance

# from pyphonetics import RefinedSoundex, Metaphone
from metaphone import doublemetaphone

from tqdm import tqdm

# to speed up, greedily assume that words with length difference of 10 probably wont be related
LEV_DIFF_CUTOFF = 5


def generate_dataset(args):

    # lemma dict -> tuple(sensekeys, gloss) from GlossBERT
    lemma_dict = defaultdict(set)
    for line in open("GlossBERT/wordnet/index.sense.gloss"):
        line = line.strip().split("\t")
        id = line[0]
        lemma_dict[id[: id.index("%")]].add((id, line[-1]))

    files_type = ["homographic", "heterographic"] if args.type == "all" else [args.type]
    for file_type in files_type:
        print(file_type)

        ph_keys = (
            {key: doublemetaphone(key)[0] for key in lemma_dict}
            if file_type == "heterographic"
            else None
        )

        output_dict = {
            "target_id": [],
            "label": [],
            "sentence": [],
            "gloss": [],
            "target_index_start": [],
            "target_index_end": [],
            "sense_key": [],
        }

        # read gold key to create label
        gold_dict = dict()
        for line in open(
            os.path.join(args.dataset_dir, "subtask3-{}-test.gold".format(file_type))
        ):
            line = line.strip().split("\t")
            gold_dict[line[0]] = set([line[1], line[2]])

        # read .data.xml
        tree = ET.parse(
            os.path.join(
                args.dataset_dir, "subtask3-{}-test.data.xml".format(file_type)
            )
        )
        root = tree.getroot()
        # read in each sentence and process at the end
        for sent in tqdm(root):
            lemma, target_id, target_index, sentence = None, None, None, []

            for i, word in enumerate(sent):
                sentence.append(word.text)
                if word.tag == "instance":
                    lemma = word.attrib["lemma"]
                    target_id = word.attrib["id"]
                    target_index = i

            assert (
                target_id is not None and lemma is not None and target_index is not None
            )

            for sense_key, gloss in get_candidates(
                lemma, file_type, lemma_dict, ph_keys
            ):
                label = 1 if sense_key in gold_dict[target_id] else 0

                output_dict["target_id"].append(target_id)
                output_dict["label"].append(label)
                output_dict["sentence"].append(" ".join(sentence))
                output_dict["gloss"].append(gloss)
                output_dict["target_index_start"].append(target_index)
                output_dict["target_index_end"].append(target_index + 1)
                output_dict["sense_key"].append(sense_key)

        df = pd.DataFrame(output_dict)
        df.to_csv(
            os.path.join(
                args.dataset_dir,
                "subtask3-{}-test_sent_cls_ws.csv".format(file_type),
            ),
            sep="\t",
            index=False,
        )


def get_candidates(lemma, file_type, lemma_dict, ph_keys):
    # candidates include all sensekey for the current lemma
    candidates = lemma_dict[lemma]

    # for heterographic, we also include intersection of word_lev and ph_lev
    if file_type == "heterographic":

        # for heterographic, prepare word_lev_dict and ph_lev_dict
        # each dict consists of key (distance: int) -> list

        word_lev_dict = defaultdict(list)
        for key in lemma_dict:
            if lemma != key and abs(len(key) - len(lemma)) < LEV_DIFF_CUTOFF:
                word_lev_dict[distance(lemma, key)].append(key)

        # ph = Metaphone()
        ph_lev_dict = defaultdict(list)
        # lemma_ph = ph.phonetics(lemma)
        lemma_ph = ph_keys[lemma]
        for key, key_ph in ph_keys.items():
            if (
                lemma_ph != key_ph
                and abs(len(key_ph) - len(lemma_ph)) < LEV_DIFF_CUTOFF
            ):
                ph_lev_dict[distance(lemma_ph, key_ph)].append(key)

        word_smallest_dist = sorted(word_lev_dict.keys())[0]
        word_lev = set(word_lev_dict[word_smallest_dist])
        phon_smallest_dist = sorted(ph_lev_dict.keys())[0]
        phon_lev = set(ph_lev_dict[phon_smallest_dist])
        for lem in word_lev.intersection(word_lev, phon_lev):
            candidates.union(lemma_dict[lem])
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, help="directory to SemEval-2017 Task 7 dataset"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "homographic", "heterographic"],
    )
    args = parser.parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()

import os
import argparse
import string
from stanza.server import CoreNLPClient
import xml.etree.ElementTree as ET


def get_upos(treebank_tag):
    if treebank_tag.startswith("N"):
        return "NOUN"
    elif treebank_tag.startswith("V") or treebank_tag in set(["MD"]):
        return "VERB"
    elif treebank_tag.startswith("J"):
        return "ADJ"
    elif treebank_tag.startswith("RB") or treebank_tag in set(["WRB"]):
        return "ADV"
    elif treebank_tag in set(["PP", "PPZ", "PRP", "PRP$", "WP"]):
        return "PRON"
    elif treebank_tag.startswith("DT") or treebank_tag in set(["WDT", "PDT"]):
        return "DET"
    elif treebank_tag.startswith("IN"):
        return "ADP"
    elif treebank_tag.startswith("CD"):
        return "NUM"
    elif treebank_tag.startswith("CC"):
        return "CONJ"
    elif treebank_tag in set(["POS", "RP", "TO", "WP$"]):
        return "PRT"
    elif treebank_tag in set(["HYPH", "SYM"]) or all(
        [c in string.punctuation for c in treebank_tag]
    ):
        return "."
    else:
        return "NOUN"


def transform_dataset(args):
    files_type = ["homographic", "heterographic"] if args.type == "all" else [args.type]
    for file_type in files_type:
        with CoreNLPClient(
            annotators=["tokenize", "pos", "lemma"],
            properties={
                "tokenize.language": "Whitespace",
                "ssplit.isOneSentence": True,
            },
            timeout=30000,
            memory="16G",
            be_quiet=True,
        ) as client:
            tree = ET.parse(
                os.path.join(args.dataset_dir, "subtask3-{}-test.xml".format(file_type))
            )
            root = tree.getroot()
            for text in root:
                text.tag = "sentence"

                # annotate to get pos and lemma
                sentence = " ".join([word.text for word in text])
                sentence_ann = client.annotate(sentence).sentence[0]

                for word, tok in zip(text, sentence_ann.token):
                    assert tok.word == word.text, "{} {}".format(tok.word, word.text)

                    if not word.text.strip():
                        continue

                    pos = tok.pos

                    lem = tok.lemma

                    if word.attrib["senses"] == "1":  # words not to evaluate
                        word.tag = "wf"
                        word.attrib = {"lemma": lem, "pos": get_upos(pos)}
                    else:  # words to evaluate
                        word.tag = "instance"
                        word.attrib = {
                            "id": word.attrib["id"],
                            "lemma": lem,
                            "pos": get_upos(pos),
                        }
            tree.write(
                os.path.join(
                    args.dataset_dir, "subtask3-{}-test.data.xml".format(file_type)
                )
            )


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
    transform_dataset(args)


if __name__ == "__main__":
    main()
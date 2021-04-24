import os
import argparse
import pandas as pd
from collections import defaultdict


def transform_result(args):
    df = pd.read_csv(args.dataset_file, sep="\t")
    result = [
        line.strip().split()
        for line in open(os.path.join(args.output_dir, "results.txt"))
    ]
    assert len(df) == len(result)

    # for each target_id, put all candidates in a list
    res_dict = defaultdict(list)
    for i, row in df.iterrows():
        res_dict[row["target_id"]].append(
            (row["sense_key"], result[i][0], result[i][-1])
        )

    with open(os.path.join(args.output_dir, "prediction.txt"), "w") as f:
        for id in res_dict:
            v = res_dict[id]
            # get the top 2 sorted by its label(1) and then score
            sorted_v = sorted(v, key=lambda x: (x[1], x[2]), reverse=True)
            if len(sorted_v) == 1:
                f.write("{} {} {}\n".format(id, sorted_v[0][0], sorted_v[0][0]))
            else:
                f.write("{} {} {}\n".format(id, sorted_v[0][0], sorted_v[1][0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, help="csv dataset used")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    transform_result(args)


if __name__ == "__main__":
    main()

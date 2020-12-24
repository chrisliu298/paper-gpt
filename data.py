import pandas as pd
from tqdm.notebook import tqdm

years = list(range(1993, 2020))

path = "/path/to/data"
dfs = [pd.read_csv(path + f"{year}.tsv", delimiter="\t") for year in tqdm(years)]
df = pd.concat(dfs)

subdf = df[["abstract", "title"]]

train_size = int(1635941 * 0.5) + 1
test_size = int(1635941 * 0.5)
print(train_size, test_size)
assert train_size + test_size == 1635941

train_df = subdf.iloc[:train_size]
test_df = subdf.iloc[train_size:]


START_TOKEN = "<|startoftext|>"
SEP_TOKEN = "<|sep|>"
END_TOKEN = "<|endoftext|>"


def make_dataset(train_df, test_df):
    with open("arxiv_train.txt", "w+") as train:
        for idx, row in tqdm(train_df.iterrows()):
            example = (
                START_TOKEN
                + " "
                + row["title"]
                + " "
                + SEP_TOKEN
                + " "
                + row["abstract"]
                + " "
                + END_TOKEN
                + "\n"
            )
            train.write(example)
        train.close()

    with open("arxiv_test.txt", "w+") as test:
        for idx, row in tqdm(test_df.iterrows()):
            example = (
                START_TOKEN
                + " "
                + row["title"]
                + " "
                + SEP_TOKEN
                + " "
                + row["abstract"]
                + " "
                + END_TOKEN
                + "\n"
            )
            test.write(example)
        test.close()


make_dataset(train_df, test_df)
import pandas as pd


def get_charset(df: pd.DataFrame) -> str:
    char_set = set()
    for _, row in df.iterrows():
        char_set = char_set | set(str(row["text"]))
    return "".join(sorted(list(char_set)))

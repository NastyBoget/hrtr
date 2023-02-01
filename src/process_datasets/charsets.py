
def check_valid_label(label: str, char_set: str) -> bool:
    return set(label).issubset(char_set)

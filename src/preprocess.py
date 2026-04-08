import csv
import re
from typing import List, Tuple


def normalize_log(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)
    text = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<TIME>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", text)
    text = re.sub(r"\b\d+\b", "<NUM>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_csv_dataset(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(normalize_log(row["text"]))
            labels.append(int(row["label"]))

    return texts, labels
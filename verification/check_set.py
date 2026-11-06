"""Part 2, Day 1: sanity checks for verification/faithfulness_set.json.
  1. counts per label (and per drafted_by)
  2. every question has exactly one answer of each label
  3. supported answers that are still empty (for you to write)
  4. items that are not checked yet
  5. supported answers that share no content word with their evidence quote (likely not grounded)
Run from the project folder: python verification/check_set.py"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

LABELS = ["supported", "contradicted", "unsupported"]
# Common words that would make almost any two sentences "overlap", so we ignore them
STOPWORDS = set("""a an the and or but of to in on at by for with from as is are was were be been being it its
this that these those than then so such which who what when where why how do does did not no can could would
should will they their them we our us he she his her you your i about into over only also very more most""".split())


def content_words(text):
    """Lowercase words without punctuation (same rule as Part 1's normalize_words), minus stopwords."""
    text = unicodedata.normalize("NFKC", text)
    words = (re.sub(r"\W+", "", w.lower()) for w in text.split())
    return {w for w in words if w and w not in STOPWORDS}


with open("verification/faithfulness_set.json", encoding="utf-8") as f:
    items = json.load(f)
with open("eval/questions.json", encoding="utf-8") as f:
    questions = {q["id"]: q for q in json.load(f)}

print(f"{len(items)} items for {len({i['question_id'] for i in items})} questions")
print("Per label:     ", dict(Counter(i["label"] for i in items)))
print("Per drafted_by:", dict(Counter(i["drafted_by"] for i in items)))
print("Per label and drafted_by:", dict(Counter(f"{i['label']}/{i['drafted_by']}" for i in items)))

# Each question should have exactly one answer per label
problems = []
for qid in questions:
    labels = Counter(i["label"] for i in items if i["question_id"] == qid)
    if labels != Counter(LABELS):
        problems.append(f"  {qid}: {dict(labels)}")
print(f"\nQuestions without exactly one answer per label: {len(problems)}")
print("\n".join(problems))

empty = [i for i in items if not i["answer"].strip()]
print(f"\nEmpty answers (still to write): {len(empty)}")
for i in empty:
    print(f"  {i['id']:18} {questions[i['question_id']]['question']}")

unchecked = [i for i in items if not i["checked"]]
print(f"\nUnchecked items: {len(unchecked)} / {len(items)}")
for i in unchecked:
    preview = i["answer"][:80] + ("..." if len(i["answer"]) > 80 else "") if i["answer"] else "(empty)"
    print(f"  {i['id']:18} [{i['drafted_by']:5}] {preview}")

# A supported answer must come from its evidence, so it should share at least one content word with it
no_overlap = []
for i in items:
    if i["label"] != "supported" or not i["answer"].strip():
        continue
    evidence = " ".join(ev["text"] for ev in questions[i["question_id"]]["evidence"])
    if not content_words(i["answer"]) & content_words(evidence):
        no_overlap.append(i)
print(f"\nSupported answers with no content-word overlap with their evidence: {len(no_overlap)}")
for i in no_overlap:
    print(f"  {i['id']:18} {i['answer']}")

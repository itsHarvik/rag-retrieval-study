"""Part 2, Day 1: build the faithfulness test set (verification/faithfulness_set.json).
For every question there are 3 answers:
  supported    - correct, and based only on the evidence quote
  contradicted - the supported answer with one fact changed, so the evidence contradicts it
  unsupported  - sounds plausible, but is not stated anywhere in the paper
The supported answers for the 15 human questions (h01-h15) are left empty for the human to write.
Run from the project folder: python verification/draft_answers.py
It refuses to overwrite an existing file (so hand-written answers can't be lost); use --force to overwrite."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

OUT_PATH = Path("verification/faithfulness_set.json")
LABELS = ["supported", "contradicted", "unsupported"]

# question id -> (supported, contradicted, unsupported). None = left for the human to write.
DRAFTS = {
    "q03": ("A cross-encoder is unsuitable for many pair regression tasks because there are too many possible sentence combinations to compare.",
            "A cross-encoder is suitable for most pair regression tasks because the number of possible sentence combinations stays small.",
            "A cross-encoder is impractical because its memory use grows quadratically with the length of each sentence pair."),
    "q04": ("With BERT, answering a single query against all Quora questions would take over 50 hours.",
            "With BERT, answering a single query against all Quora questions would take over 5 hours.",
            "With BERT, answering a single Quora query would cost roughly 2,000 US dollars of cloud GPU time."),
    "q05": ("No. Averaging BERT's outputs gives rather bad sentence embeddings, often worse than averaged GloVe embeddings, with an average correlation of only 54.81.",
            "No. Averaging BERT's outputs gives rather bad sentence embeddings, often worse than averaged GloVe embeddings, with an average correlation of only 64.81.",
            "No. Averaged BERT vectors are dominated by a few very frequent words, so most sentences end up with nearly identical embeddings."),
    "q06": ("SBERT was fine-tuned on NLI (natural language inference) data, including the SNLI dataset.",
            "SBERT was fine-tuned on Reddit conversation data instead of NLI data such as SNLI.",
            "SBERT was trained on a corpus of 2 million paraphrase pairs that the authors mined from news headlines."),
    "q07": ("The SNLI dataset contains 570,000 sentence pairs.",
            "The SNLI dataset contains 430,000 sentence pairs.",
            "The SNLI sentence pairs were written by about 2,500 crowd workers on Amazon Mechanical Turk."),
    "q08": ("Training used a batch size of 16 and the Adam optimizer with a learning rate of 2e-5.",
            "Training used a batch size of 32 and the Adam optimizer with a learning rate of 2e-5.",
            "Training used gradient clipping at 1.0 and a weight decay of 0.01."),
    "q09": ("SBERT experiments with three pooling strategies.",
            "SBERT experiments with two pooling strategies.",
            "SBERT experiments with attention-based pooling, where a small learned network weights the output of each token."),
    "q11": ("The sentence embeddings u and v are concatenated with their element-wise difference |u−v|.",
            "The sentence embeddings u and v are concatenated with their element-wise product u∗v.",
            "The two sentence vectors are first passed through a shared tanh layer and then averaged before the softmax."),
    "q12": ("The regression objective uses mean-squared-error loss.",
            "The regression objective uses cross-entropy loss.",
            "The regression objective uses a contrastive loss with in-batch negatives."),
    "q13": ("Triplet training tunes the network so that the anchor is closer to the positive example than to the negative example.",
            "Triplet training tunes the network so that the anchor is closer to the negative example than to the positive example.",
            "Triplet training makes the embeddings robust to spelling errors by using a misspelled copy of the anchor as the positive."),
    "q14": ("The margin ϵ is set to 1.",
            "The margin ϵ is set to 0.5.",
            "The margin is increased gradually from 0.1 during the first training epoch."),
    "q16": ("Averaged BERT embeddings achieve an average correlation of only 54.81.",
            "Averaged BERT embeddings achieve an average correlation of only 61.32.",
            "Averaged BERT embeddings improve by about 10 points when stop words are removed before averaging."),
    "q17": ("Because Universal Sentence Encoder was trained on various datasets, including news, question-answer pages and discussion forums.",
            "Because Universal Sentence Encoder was trained only on news articles.",
            "Because SICK-R sentences are much longer than the NLI sentences SBERT was trained on."),
    "q18": ("No. The authors observe only a minor difference between SBERT and SRoBERTa, and no significant improvement from RoBERTa.",
            "Yes. Replacing BERT with RoBERTa gives a significant improvement over SBERT.",
            "No, but SRoBERTa trains about twice as fast as SBERT."),
    "q19": ("The STS benchmark contains 8,628 sentence pairs.",
            "The STS benchmark contains 5,749 sentence pairs.",
            "The STS benchmark contains sentence pairs in both English and Spanish."),
    "q21": ("The AFS corpus covers gun control, gay marriage, and the death penalty.",
            "The AFS corpus covers gun control, abortion, and the death penalty.",
            "The AFS corpus covers political topics taken from US presidential election debates."),
    "q22": ("When tested on an unseen topic, SBERT drops by about 7 points Spearman correlation.",
            "When tested on an unseen topic, SBERT drops by about 17 points Spearman correlation.",
            "On unseen topics, SBERT's performance falls the most on the death penalty topic."),
    "q23": ("Because BERT is able to use attention to compare both sentences directly.",
            "Because BERT is not able to use attention to compare both sentences directly, while SBERT is.",
            "Because argument sentences in AFS are often sarcastic, which sentence embeddings struggle to capture."),
    "q24": ("The anchor and the positive example come from the same Wikipedia section, while the negative example comes from a different section of the same article.",
            "The anchor and the positive example come from the same Wikipedia section, while the negative example comes from a different article.",
            "Each triplet pairs a sentence with the sentence directly after it as the positive example."),
    "q25": ("SBERT reaches an accuracy of 0.8042 (base) and 0.8078 (large), clearly outperforming the BiLSTM approach by Dor et al.",
            "SBERT reaches an accuracy of 0.7042 (base) and 0.8078 (large), clearly outperforming the BiLSTM approach by Dor et al.",
            "SBERT reaches about the same accuracy as human annotators on the Wikipedia section triplets."),
    "q26": ("Seven SentEval transfer tasks are used for evaluation.",
            "Five SentEval transfer tasks are used for evaluation.",
            "The SentEval tasks used include natural language inference on SICK-E and image-caption retrieval on COCO."),
    "q28": ("Because cosine similarity treats all dimensions of the embedding equally.",
            "Because cosine similarity gives more weight to the first dimensions of the embedding.",
            "Because BERT embeddings all point in a narrow cone, so almost every cosine score is close to 1."),
    "q29": ("The element-wise difference |u−v| is the most important component.",
            "The element-wise product u∗v is the most important component.",
            "The vector u of the first sentence matters most, because it acts as the query."),
    "q30": ("The MAX strategy performs significantly worse than MEAN or CLS-token pooling.",
            "The CLS-token strategy performs significantly worse than MEAN or MAX pooling.",
            "Mean pooling over only the last four layers performs worst."),
    "q32": ("On a GPU, SBERT with smart batching is about 9% faster than InferSent.",
            "On a GPU, SBERT with smart batching is about 55% faster than InferSent.",
            "On a GPU, SBERT is about twice as fast as InferSent when the batch size is above 256."),
    "q33": ("Because InferSent uses a single BiLSTM layer, while BERT uses 12 stacked transformer layers.",
            "Because InferSent uses a single BiLSTM layer, while BERT uses 24 stacked transformer layers.",
            "Because InferSent's code uses multi-threaded CPU kernels that BERT does not support."),
    "q34": ("The research was supported by the German Research Foundation.",
            "The research was supported by the European Research Council.",
            "The research was funded by a Google Faculty Research Award."),
    "q35": ("The speed measurements ran on a server with an Intel i7-5820K CPU @ 3.30GHz and an Nvidia Tesla GPU.",
            "The speed measurements ran on a server with an Intel i9-9900K CPU @ 3.30GHz and an Nvidia Tesla GPU.",
            "The speed measurements ran on a server with 64 GB of RAM, using 8 CPU threads."),
    # Human-written questions: supported answer left empty (None) for the human to write.
    "h01": (None,
            "Finding the most similar pair among 10,000 sentences with BERT requires about 5 million (4,999,500) inference computations.",
            "With BERT, the number of inference computations can be halved by caching the encoding of the first sentence."),
    "h02": (None,
            "SBERT reduces the effort from 65 hours to about 5 minutes.",
            "SBERT makes the search so fast that it can run in real time on a smartphone."),
    "h03": (None,
            "SBERT improves over InferSent by 5.5 points on the seven STS tasks.",
            "SBERT's improvement over InferSent comes mainly from its larger vocabulary."),
    "h04": (None,
            "The default pooling strategy for SBERT is CLS.",
            "By default, SBERT pools the outputs while ignoring padding and special tokens such as [SEP]."),
    "h05": (None,
            "SBERT can be tuned in less than 20 hours.",
            "Tuning time depends mainly on sentence length, which the authors capped at 128 tokens."),
    "h06": (None,
            "The SNLI pairs are annotated with the labels contradiction, entailment, and paraphrase.",
            "Each SNLI label was assigned by a majority vote of five annotators."),
    "h07": (None,
            "The AFS corpus annotated 4,000 sentential argument pairs from social media dialogs.",
            "The argument pairs in the AFS corpus were collected from the website CreateDebate."),
    "h08": (None,
            "Dor et al. fine-tuned a CNN architecture with triplet loss to derive sentence embeddings.",
            "Dor et al. initialised their network with pre-trained word2vec vectors before triplet training."),
    "h09": (None,
            "The only dataset where SBERT is significantly worse than Universal Sentence Encoder is MRPC.",
            "SBERT is significantly worse than Universal Sentence Encoder on multilingual datasets."),
    "h10": (None,
            "On CPU, InferSent is about 55% faster than SBERT.",
            "On CPU, InferSent's speed advantage disappears for sentences longer than 50 words."),
    "h11": (None,
            "Because Spearman's rank correlation is badly suited for STS, the authors compute the Pearson correlation between the cosine similarity of the embeddings and the gold labels.",
            "Because Spearman correlation is required by the official SemEval scoring script."),
    "h12": (None,
            "First training on NLI and then on STSb leads to a slight improvement of 3-4 points.",
            "Because NLI pretraining teaches SBERT to ignore word order, which the STS benchmark rarely depends on."),
    "h13": (None,
            "Because the siamese network derives variable-length vectors for the input sentences, and similarity measures on them can be computed extremely efficiently on modern hardware.",
            "Because SBERT compresses each sentence into a 64-dimensional binary code, so comparisons only need bit operations."),
    "h14": (None,
            "Because BERT's construction makes it unsuitable for semantic similarity search, even though averaging its outputs gives good sentence embeddings.",
            "Because the original BERT model only supports English, while SBERT supports over 50 languages."),
    "h15": (None,
            "Sentences with similar lengths are grouped together and padded to the longest sentence in the whole dataset, which reduces the overhead from padding tokens.",
            "Smart batching sorts sentences alphabetically so that repeated words can be cached and reused."),
}


def make_items(questions):
    items = []
    for q in questions:
        answers = DRAFTS[q["id"]]  # KeyError here means a question has no drafts yet
        for label, answer in zip(LABELS, answers):
            items.append({
                "id": f"{q['id']}-{label}",
                "question_id": q["id"],
                "answer": answer or "",
                "label": label,
                # Only the empty supported answers of the human questions are for the human to write
                "drafted_by": "human" if answer is None else "llm",
                "checked": False,
            })
    return items


if __name__ == "__main__":
    if OUT_PATH.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT_PATH} already exists (it may contain your hand-written answers). "
                 "Run with --force to overwrite it.")
    with open("eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    extra = set(DRAFTS) - {q["id"] for q in questions}
    if extra:
        sys.exit(f"Drafts for unknown question ids: {sorted(extra)}")

    items = make_items(questions)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    to_write = sum(i["answer"] == "" for i in items)
    print(f"Saved {len(items)} items for {len(questions)} questions to {OUT_PATH}")
    print(f"{to_write} supported answers are empty and waiting for you to write them")

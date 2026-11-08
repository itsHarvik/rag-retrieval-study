# Review log: faithfulness_set.json

Review of the 114 LLM-drafted items (2026-11-04 to 2026-11-06, with a re-review of the 72 bulk-approved items on 2026-11-07), using the groups in `precheck.md`.
The 15 human-written supported answers (h01–h15) were written and checked by the author before this review.

## Summary

| Decision | Count |
|---|---|
| fix | 10 |
| drop | 0 |
| ok | 104 |

Final state after `python verification/check_set.py`: 129 items (43 per label), 0 unchecked, 0 empty, 0 supported answers without content-word overlap with their evidence.

## Fixes

The first five fixes used the text suggested in `precheck.md` (group A); the five marked "re-review" came from the 2026-11-07 re-review. Each fixed item keeps `drafted_by: "llm"` and has `edited_by_author: true`.

### q09-supported
- Before: SBERT experiments with three pooling strategies.
- After: SBERT experiments with three pooling strategies: the output of the CLS token, the mean of all output vectors (MEAN), and a max-over-time of the output vectors (MAX).

### q11-contradicted
- Before: The sentence embeddings u and v are concatenated with their element-wise product u∗v.
- After: The sentence embeddings u and v are concatenated with their element-wise sum u+v.

### q26-supported
- Before: Seven SentEval transfer tasks are used for evaluation.
- After: Seven SentEval transfer tasks are used: MR, CR, SUBJ, MPQA, SST, TREC and MRPC.

### q26-unsupported
- Before: The SentEval tasks used include natural language inference on SICK-E and image-caption retrieval on COCO.
- After: The seven SentEval tasks were chosen by surveying NLP practitioners about which tasks matter most.

### q33-contradicted
- Before: Because InferSent uses a single BiLSTM layer, while BERT uses 24 stacked transformer layers.
- After: Because InferSent uses a single BiLSTM layer, while BERT uses 6 stacked transformer layers.

### q17-contradicted (re-review, 2026-11-07)
- Before: Because Universal Sentence Encoder was trained only on news articles.
- After: Because Universal Sentence Encoder was trained only on NLI data, the same data as SBERT.
- Why: the old answer changed only one word ("only") and shared "news" with the evidence, so it barely differed from the supported answer. Text suggested by the LLM, chosen by the author.

### q29-unsupported (re-review, 2026-11-07)
- Before: The vector u of the first sentence matters most, because it acts as the query.
- After: The element-wise difference |u−v| matters most, and the authors plan to replace it with a learned distance function in future work.
- Why: the question asks which part matters most and p.8 names |u−v|, so naming another part was really contradicted. The new answer keeps the true part and adds a claim the paper never makes (the paper has no "future work" text; checked by searching it). Now a mixed answer, like q18-unsupported.

### q30-unsupported (re-review, 2026-11-07)
- Before: Mean pooling over only the last four layers performs worst.
- After: The MAX strategy performs worst, and the authors plan to test attention-based pooling in future work.
- Why: p.8 says MAX performs worst, so naming another strategy as worst was really contradicted. Same kind of fix as q29-unsupported (paper searched for "future" and "attention-based": no hits).

### q34-contradicted (re-review, 2026-11-07)
- Before: The research was supported by the European Research Council.
- After: The research received no external funding.
- Why: the old answer had the same shape as q34-unsupported (a funder the paper never mentions), so its label was inconsistent. The new answer clearly conflicts with p.9 ("supported by the German Research Foundation… co-funded by the German Federal Ministry of Education and Research (BMBF)").

### h09-unsupported (re-review, 2026-11-07)
- Before: SBERT is significantly worse than Universal Sentence Encoder on multilingual datasets.
- After: SBERT is significantly worse only on TREC, and the authors plan to close this gap in future work by adding question-answering data to SBERT's training.
- Why: p.7 says TREC is "the only dataset" where SBERT is significantly worse, so naming another dataset was really contradicted. The new answer gives no reason for the gap, because p.7 already gives one (USE was pre-trained on question-answering data); it adds a future-work plan instead, which the paper never mentions.

## Drops

None.

## Items marked ok

- Group B (tricky on purpose; shown one by one): 7 items: q07-contradicted, q16-contradicted, q19-contradicted, q32-contradicted, q12-contradicted, q06-contradicted, h11-unsupported.
- Group C, batch 1 (shown with question and evidence): 15 items (q03–q08).
- Group C, batch 2 (shown with question and evidence): 15 items (q08–q16).
- Group C, remaining: 72 items first approved in bulk ("all ok") without being displayed. Re-reviewed on 2026-11-07: shown to the author in 4 batches of 18 (id, label, answer, evidence quote), with any open notes shown next to their item.
  - Re-review batch 1 (q17–q24, 18 items): 17 ok, 1 fixed (q17-contradicted). The author considered changing all three topics in q21-contradicted and kept it as it is: a single wrong topic is a more realistic hallucination and matches the other contradicted items, which change one fact each. q18-unsupported and q21-unsupported were shown with their open notes and kept.
  - Re-review batch 2 (q24–q33, 18 items): 16 ok, 2 fixed (q29-unsupported, q30-unsupported, which resolves their open notes). q32-unsupported (arguably contradicted by the 9% figure) and q25-contradicted (one changed digit) were discussed and kept.
  - Re-review batch 3 (q33–h06, 18 items): 17 ok, 1 fixed (q34-contradicted, which resolves its open note). h03-contradicted was shown as tricky on purpose (5.5 is SBERT's gain over Universal Sentence Encoder on p.2, not over InferSent) and kept.
  - Re-review batch 4 (h06–h15, 18 items): 17 ok, 1 fixed (h09-unsupported, which resolves its open note). h10-contradicted (55% is SBERT vs USE on GPU, p.8) and h12-contradicted (3-4 points is the BERT cross-encoder's gain, p.5) were shown as tricky on purpose and kept.
  - Re-review total: 72 items shown, 67 ok, 5 fixed, 0 dropped. Every item in faithfulness_set.json has now been shown to the author.
  - For the unsupported items, the LLM searched the full paper text for each claim's key terms before showing the batch. All five fixes were suggested by the LLM and chosen by the author.

## Open notes (not decisions)

Possible label issues that the LLM noted. Unless marked resolved, they were approved as they are. They're listed here so they can be revisited or reported as a limitation.

**Unsupported claims that arguably clash with the paper (could count as contradicted):**

| id | clashes with |
|---|---|
| q09-unsupported | p.3 names exactly three pooling strategies (CLS, MEAN, MAX) |
| q11-unsupported | p.3 "We concatenate…" (the claim says "averaged") |
| q12-unsupported | p.3 the regression loss is MSE |
| q14-unsupported | p.3 "we set ε = 1" |
| q21-unsupported* | p.5 the pairs come "from social media dialogs" (the claim says presidential debates) |
| ~~q29-unsupported~~ | resolved: fixed in the 2026-11-07 re-review |
| ~~q30-unsupported~~ | resolved: fixed in the 2026-11-07 re-review |
| ~~h09-unsupported~~ | resolved: fixed in the 2026-11-07 re-review |
| h11-unsupported | p.4 the real reason is that Pearson is badly suited for STS |

\* = in the formerly bulk-approved set; shown to the author with this note in the 2026-11-07 re-review and kept.

**Other notes:**

- q04-contradicted: "over 5 hours" is technically still true if the real time is over 50 hours.
- q03-contradicted: flips two things, not one.
- ~~q34-contradicted~~: resolved, fixed in the 2026-11-07 re-review.
- q05-unsupported, q18-unsupported*: mixed answers (the opening "No" is supported; the reason is made up).
- q29-unsupported, q30-unsupported, h09-unsupported (after the fix): also mixed answers (true first half, invented future-work plan).
- q06-supported: leaves out Multi-Genre NLI.
- q09-supported, q26-supported (after the fix): use the full sentence, which goes beyond the evidence quote stored in `eval/questions.json`.

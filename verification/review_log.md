# Review log: faithfulness_set.json

Review of the 114 LLM-drafted items (2026-11-04 to 2026-11-06), using the groups in `precheck.md`.
The 15 human-written supported answers (h01–h15) were written and checked by the author before this review.

## Summary

| Decision | Count |
|---|---|
| fix | 5 |
| drop | 0 |
| ok | 109 |

Final state after `python verification/check_set.py`: 129 items (43 per label), 0 unchecked, 0 empty, 0 supported answers without content-word overlap with their evidence.

## Fixes

All fixes used the text suggested in `precheck.md` (group A). Each fixed item keeps `drafted_by: "llm"` and has `edited_by_author: true`.

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

## Drops

None.

## Items marked ok

- Group B (tricky on purpose; shown one by one): 7 items: q07-contradicted, q16-contradicted, q19-contradicted, q32-contradicted, q12-contradicted, q06-contradicted, h11-unsupported.
- Group C, batch 1 (shown with question and evidence): 15 items (q03–q08).
- Group C, batch 2 (shown with question and evidence): 15 items (q08–q16).
- Group C, remaining: 72 items approved in bulk ("all ok") **without being displayed one by one**. They were checked against the paper text by the LLM only (see below).

## Open notes (not decisions)

Possible label issues that the LLM noted. All were approved as they are. They're listed here so they can be revisited or reported as a limitation.

**Unsupported claims that arguably clash with the paper (could count as contradicted):**

| id | clashes with |
|---|---|
| q09-unsupported | p.3 names exactly three pooling strategies (CLS, MEAN, MAX) |
| q11-unsupported | p.3 "We concatenate…" (the claim says "averaged") |
| q12-unsupported | p.3 the regression loss is MSE |
| q14-unsupported | p.3 "we set ε = 1" |
| q21-unsupported* | p.5 the pairs come "from social media dialogs" (the claim says presidential debates) |
| q29-unsupported* | p.8 "The most important component is the element-wise difference \|u −v\|" |
| q30-unsupported* | p.8 "the MAX strategy perform significantly worse" |
| h09-unsupported* | p.7 "The only dataset where SBERT is significantly worse… is the TREC dataset" |
| h11-unsupported | p.4 the real reason is that Pearson is badly suited for STS |

\* = in the bulk-approved set, never shown to the author.

**Other notes:**

- q04-contradicted: "over 5 hours" is technically still true if the real time is over 50 hours.
- q03-contradicted: flips two things, not one.
- q34-contradicted*: the paper names two funders (DFG and BMBF); "European Research Council" is not explicitly ruled out, so it is arguably unsupported.
- q05-unsupported, q18-unsupported*: mixed answers (the opening "No" is supported; the reason is made up).
- q06-supported: leaves out Multi-Genre NLI.
- q09-supported, q26-supported (after the fix): use the full sentence, which goes beyond the evidence quote stored in `eval/questions.json`.

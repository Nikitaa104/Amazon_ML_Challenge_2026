# output/ — Submission Output Files

This folder will be populated after the full-scale training run completes.

## Expected files after `python -m src.main --mode predict`:

- `candidate_pairs.tsv`   — All (source1_entity_id, candidate_entity_id) pairs considered
- `matching_results.tsv`  — Final predicted matches after F0.5-tuned threshold filtering

## TODO
- [ ] Run `python -m src.main --mode predict` after training completes
- [ ] Verify both files pass `python -m src.main --mode validate`
- [ ] Copy files here for submission packaging

Analyze the latest training run and recommend next steps.

Do this:

1. Read `training_guide.md` for context on goals and competition constraints
2. Read `training_status.json` to see what just finished
3. Read the full metrics log at `logs/<run_name>/metrics.jsonl` — look at loss trends across all iterations
4. Read `logs/<run_name>/eval.jsonl` if it exists — check win rates vs baselines
5. Read `training_journal.md` to see what was tried in previous runs
6. Read the config file that was used (from training_status.json → config_file)

Then:

7. Analyze: Are losses decreasing? Have they plateaued? How are win rates? What changed vs previous runs?
8. Decide what to do next: continue same phase, advance to next phase, tune parameters, or declare done
9. If config changes are needed, make them (archive the old config first)
10. Write a journal entry to `training_journal.md` with your analysis, decision, and the exact command for the next run
11. Tell me the command to run

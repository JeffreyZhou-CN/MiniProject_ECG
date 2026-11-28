# AI usage log

Tool: Claude (Anthropic), used as a step-by-step assistant following the module's guidelines on
agentic AI tools: small tasks, tests as acceptance criteria, human review before every commit.

| Date | Step | What the AI produced | What I did / checked |
|---|---|---|---|
| 2026-11-25 | 0. Set-up | Drafts of `.gitignore`, `requirements.txt`, `environment.yml`, `docs/conventions.md`, the README introduction and the first progress-log entry | Created each file myself, checked the environment definition, read every line, shortened and edited the README and progress log |
| 2026-11-25 | 1. Data generator | `src/data.py` (course-guide generator with docstring and input check), `pytest.ini`, `tests/test_data.py` (6 tests) | Read the code line by line, reformatted the docstring, ran the tests on my laptop (6 passed); an AI-run check confirmed the output is identical to the course guide's code |
| 2026-11-25 | 2. Baseline LMS | `src/baseline.py` (`lms_filter`) and `tests/test_baseline.py` (11 tests) | Worked through the update order, the zero-reference case and the guide's hand example with the AI's explanation; ran the tests on my laptop
| 2026-11-25 | 3. NLMS modification | `src/modified.py` (`nlms_filter` with ablation switch) and `tests/test_modified.py` (14 tests), with comments | Worked through the NLMS update, the ablation idea and the hand example (NLMS vs. LMS overshoot) with the AI's explanation; asked for the reasoning behind each test case; ran the tests on my laptop. |
| 2026-11-25 | 4. Metrics | `src/metrics.py` (MSE, SNR, R-peak error, paired differences with Student-t intervals) and tests, added in three small commits | Worked through the SNR/MSE hand example, why the input SNR is needed, the R-peak metric and the paired t-interval with the AI's explanation; ran the tests on my laptop (44 passed in total) |
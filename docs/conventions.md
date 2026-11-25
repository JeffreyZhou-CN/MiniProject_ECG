# Project conventions

Rules every function in this project must follow (used when prompting the AI and when reviewing).

1. Implement the core algorithms with NumPy only; no ready-made adaptive-filter libraries.
2. Every function has a docstring stating parameters, array shapes, units and return values.
3. Functions never modify their input arrays.
4. Invalid inputs raise `ValueError` with a clear message.
5. Every function has tests: a hand-computed example, an edge case and a relevant property.
6. Randomness only through an explicit seed (`numpy.random.default_rng(seed)`).
7. One small function per change; at most ~100 lines of diff per commit.
8. Run `python -m pytest` before every commit.
9. Figures are generated from saved results by code; no numbers are typed in by hand.
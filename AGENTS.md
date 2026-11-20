## Shared Developer policy

Read `/Users/uiwwsw/Developer/AGENTS.md` at task start, including its linked policies when applicable. Shared security approvals, SSD execution/retention, protected-main PR-and-merge delivery, MOVV decisions/sync, typography, commit checks rules are inherited from there; do not duplicate them here. Run local commands through `./dev` or `devctl exec --`. Keep product-specific additions in this file.

## Nascen

- Python 3.12–3.14, CPU only. Use `.venv` inside this repo.
- Tests: `./dev .venv/bin/python -m unittest discover -s tests`.
- Published numbers in `README.md` and `docs/experiments.md` must match the JSON results in `docs/`.

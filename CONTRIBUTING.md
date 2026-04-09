# Contributing to PAMPAr-Coder

Thank you for your interest in contributing! PAMPAr-Coder is licensed under
[BUSL-1.1](LICENSE) — contributions are welcome for non-commercial research
and academic purposes.

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/lucasmella-stack/PAMPAr-Coder/issues) first.
2. Open a new issue with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version, OS, GPU (if relevant)

### Suggesting Features

Open an issue with the `enhancement` label describing the feature and its use case.

### Pull Requests

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Follow existing code style (type hints, Google-style docstrings).
4. Add tests for new functionality (`pytest`).
5. Run the test suite: `python -m pytest tests/ -v`
6. Commit with [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
7. Open a PR against `main`.

## Development Setup

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
cd PAMPAr-Coder
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Code Style

- Python 3.11+ with type hints everywhere
- Docstrings: Google style
- No hardcoded paths or secrets
- Functions ≤ 50 lines, files ≤ 400 lines

## License

By contributing, you agree that your contributions will be licensed under the
same [BUSL-1.1](LICENSE) license.

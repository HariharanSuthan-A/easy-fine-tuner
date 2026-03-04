# PyPI Publishing Guide

This guide explains how to publish easyfinetuner to PyPI so users can install it via `pip install easyfinetuner`.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org)
2. **TestPyPI Account** (optional but recommended): Create an account at [test.pypi.org](https://test.pypi.org)
3. **API Tokens**: Generate API tokens from your PyPI account settings

## Step 1: Install Build Tools

```bash
pip install --upgrade pip
pip install build twine
```

## Step 2: Update Version (Important!)

Before each new release, update the version number in:
- `setup.py` - line 15
- `pyproject.toml` - line 5

Use semantic versioning: `MAJOR.MINOR.PATCH` (e.g., `0.1.0`, `0.1.1`, `0.2.0`)

## Step 3: Build the Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build the package
python -m build
```

This creates:
- `dist/easyfinetuner-0.1.0-py3-none-any.whl` (wheel)
- `dist/easyfinetuner-0.1.0.tar.gz` (source distribution)

## Step 4: Test on TestPyPI (Optional but Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
cd /tmp
pip install --index-url https://test.pypi.org/simple/ easyfinetuner
```

## Step 5: Upload to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Enter your PyPI API token when prompted
# Username: __token__
# Password: pypi-xxxxxxxxxxxx
```

## Step 6: Verify Installation

```bash
# Test in a clean environment
cd /tmp
pip install easyfinetuner
python -c "from easyfinetuner import FineTuner; print('Success!')"
```

## Alternative: Using GitHub Actions (Automated)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

Then:
1. Go to GitHub → Settings → Secrets → New repository secret
2. Add `PYPI_API_TOKEN` with your PyPI API token
3. Create a new GitHub release to trigger automatic publishing

## Troubleshooting

### "File already exists" Error
- Version already uploaded. Increment version in `setup.py` and `pyproject.toml`

### "Invalid API Token" Error
- Make sure you're using `__token__` as username and the full token (including `pypi-` prefix) as password

### Package Not Installing Correctly
- Test locally first: `pip install dist/easyfinetuner-0.1.0-py3-none-any.whl`

### Check Package Contents
```bash
# List what's in the wheel
unzip -l dist/easyfinetuner-0.1.0-py3-none-any.whl
```

## Checklist Before Publishing

- [ ] Version updated in `setup.py` and `pyproject.toml`
- [ ] `LICENSE` file exists
- [ ] `README.md` is complete
- [ ] `requirements.txt` is up to date
- [ ] All tests pass
- [ ] Build succeeds without errors
- [ ] Package installs correctly locally

## After Publishing

1. **Tag the release** in Git:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Update GitHub releases** with changelog

3. **Verify on PyPI**: https://pypi.org/project/easyfinetuner/

4. **Test installation**:
   ```bash
   pip install easyfinetuner
   ```

## Quick Reference

```bash
# One-liner build and upload
rm -rf build/ dist/ *.egg-info && python -m build && python -m twine upload dist/*

# Build only
python -m build

# Upload only
python -m twine upload dist/*
```

---

**Your package URL after publishing**: https://pypi.org/project/easyfinetuner/

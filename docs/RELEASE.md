# Release Guide

## Setup (One-time)

Add `PYPI_API_TOKEN` to GitHub Secrets:
1. Get token: https://pypi.org/manage/account/token/
2. Add to secrets: Settings → Secrets → Actions → `PYPI_API_TOKEN`

## Release (One-click)

1. Go to: https://github.com/StarTrail-org/LEANN/actions/workflows/release-manual.yml
2. Click "Run workflow"
3. Enter version: `0.1.2`
4. Click green "Run workflow" button

That's it! The workflow will automatically:
- ✅ Update version in all packages
- ✅ Build all packages
- ✅ Publish to PyPI
- ✅ Create GitHub tag and release

Check progress: https://github.com/StarTrail-org/LEANN/actions

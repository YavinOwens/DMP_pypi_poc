# Read the Docs Setup

This package is configured for automatic documentation hosting on Read the Docs.

## Setup Instructions

1. **Sign up for Read the Docs**:
   - Go to https://readthedocs.org/
   - Sign up or log in with your GitHub account

2. **Import the Project**:
   - Click "Import a Project"
   - Select your GitHub repository: `YavinOwens/DMP_pypi_poc`
   - Name: `datamanagement-genai` (or your preferred name)
   - Click "Create"

3. **Configure Build Settings**:
   - The `.readthedocs.yaml` file is already configured
   - Read the Docs will automatically detect it
   - Python version: 3.11
   - Build command: Automatic (uses Sphinx)

4. **Build Documentation**:
   - Click "Build version" to trigger the first build
   - Documentation will be available at: `https://datamanagement-genai.readthedocs.io/`

## Documentation Structure

The documentation is built using Sphinx and includes:

- **Installation Guide**: How to install and configure the package
- **Quick Start**: Getting started examples
- **API Reference**: Complete API documentation (auto-generated)
- **Examples**: Practical usage examples
- **Contributing**: Guidelines for contributors

## Local Documentation Build

To build documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

Or using Sphinx directly:

```bash
pip install -e ".[docs]"
cd docs
sphinx-build -b html . _build/html
```

## Updating Documentation

1. Edit files in the `docs/` directory
2. Commit and push changes
3. Read the Docs will automatically rebuild the documentation

## Custom Domain (Optional)

You can set up a custom domain in Read the Docs project settings:
- Settings → Domains
- Add your custom domain (e.g., `docs.yourdomain.com`)

## Troubleshooting

If builds fail:
- Check the build logs in Read the Docs dashboard
- Ensure all dependencies are listed in `pyproject.toml` under `[project.optional-dependencies]`
- Verify `.readthedocs.yaml` syntax is correct

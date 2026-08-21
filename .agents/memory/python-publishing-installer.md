---
name: Python publishing installer
description: Deployment-specific Python package installation behavior for the Streamlit artifact.
---

Published Python services use the workspace `.pythonlibs` interpreter, but that interpreter may not include the `pip` module. Production build commands should use the available `uv` executable with `uv pip install --python /home/runner/workspace/.pythonlibs/bin/python ...` rather than `python -m pip`.

**Why:** The first production build failed before starting the app with `No module named pip`.

**How to apply:** When configuring a Python artifact's production build, install its requirements through `uv pip` and target the same `.pythonlibs` interpreter used by the runtime.
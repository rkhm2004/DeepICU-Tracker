import os
from pathlib import Path

# Define the base directory (current working directory)
base_dir = Path.cwd()

# Define the directory structure
directories = [
    "data/raw",
    "data/processed",
    "src/data_pipeline",
    "src/vae",
    "src/markov_mgf",
    "src/bayesian_net",
    "src/dashboard",
    "notebooks",
    "tests",
    "docs/report"
]

print("Creating directories and .gitkeep files...")
# Create directories and .gitkeep files
for dir_path in directories:
    path = base_dir / dir_path
    path.mkdir(parents=True, exist_ok=True)
    # Create a .gitkeep file so Git tracks empty folders
    (path / ".gitkeep").touch()
    print(f"Created: {dir_path}")

# Define file contents
gitignore_content = """# Ignore raw patient data
data/raw/

# Python caches and environments
__pycache__/
*.py[cod]
*$py.class
.env
venv/
.venv/
"""

readme_content = """# ICU Early-Warning System

Explainable Early-Warning System for ICU Patient Deterioration using a 4-stage pipeline: VAE -> CTMC -> Phase-Type MGF -> Bayesian Network.

## Team Structure
* **Member A:** Data Pipeline (MIMIC-IV)
* **Member B:** VAE (Stage 1)
* **Member C:** CTMC + Phase-Type MGF (Stages 2 & 3)
* **Member D:** Bayesian Network + Dashboard (Stage 4)

## Branching Strategy
* `main`: Production-ready code.
* `develop`: Integration branch.
* `feature/<name>`: Individual work branches.
"""

print("\nCreating base files...")

# Create .gitignore
with open(base_dir / ".gitignore", "w") as f:
    f.write(gitignore_content.strip())
print("Created: .gitignore")

# Create requirements.txt (empty)
(base_dir / "requirements.txt").touch()
print("Created: requirements.txt")

# Create README.md
with open(base_dir / "README.md", "w") as f:
    f.write(readme_content.strip())
print("Created: README.md")

print("\nProject structure and base files created successfully! You can now run your git init and push commands.")
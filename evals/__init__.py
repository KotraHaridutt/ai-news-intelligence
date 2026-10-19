"""Feature 2 evaluation tools. Run them with the project's own environment (where the app's packages
are installed), not the system Python:

  .local/venv/Scripts/python.exe evals/review_labels.py
"""
import sys

try:
    import langchain_text_splitters  # noqa: F401  (one of the app's packages: if it's missing, so are the rest)
except ImportError:
    sys.exit(f"This Python ({sys.executable}) doesn't have the app's packages.\n"
             "Run the script with the project's environment instead, from the project folder:\n"
             "  .local\\venv\\Scripts\\python.exe " + " ".join(sys.argv))

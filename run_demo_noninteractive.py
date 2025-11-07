"""Run the CLI demo non-interactively by injecting stdin."""
import sys
import io

# Ensure package imports work when run from repository root
from src.cli_demo import run_demo

def main():
    sample = "What is the recommended empiric outpatient antibiotic for community-acquired pneumonia?\n"
    sys.stdin = io.StringIO(sample)
    run_demo()

if __name__ == '__main__':
    main()

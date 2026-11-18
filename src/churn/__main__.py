"""Allow `python -m churn ...`."""

import sys

from churn.cli import main

if __name__ == "__main__":
    sys.exit(main())

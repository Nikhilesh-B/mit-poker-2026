"""Convert test_deepcfr_network.py to Jupyter notebook"""
import json

# Create notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "header",
            "metadata": {},
            "source": [
                "# DeepCFR Network Unit Tests\n",
                "\n",
                "**Comprehensive tests for the DeepCFR neural network module**\n",
                "\n",
                "Tests the network with real parsed infosets from MCCFR."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "imports",
            "metadata": {},
            "outputs": [],
            "source": [
                "from mccfr import MCCFR\n",
                "from infoset_parser import parse_infoset_to_network_input, batch_parse_infosets\n",
                "from DeepCFR import DeepCFRModule\n",
                "import numpy as np\n",
                "import random\n",
                "import torch\n",
                "import sys\n",
                "import os\n",
                "\n",
                "# Add parent directory to path\n",
                "parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))\n",
                "sys.path.insert(0, parent_dir)\n",
                "\n",
                "# Test tracking\n",
                "tests_passed = 0\n",
                "tests_failed = 0\n",
                "\n",
                "def run_test(test_name, test_func):\n",
                "    global tests_passed, tests_failed\n",
                "    try:\n",
                "        test_func()\n",
                "        print(f'✓ {test_name} PASSED')\n",
                "        tests_passed += 1\n",
                "    except AssertionError as e:\n",
                "        print(f'✗ {test_name} FAILED: {e}')\n",
                "        tests_failed += 1\n",
                "    except Exception as e:\n",
                "        print(f'✗ {test_name} ERROR: {e}')\n",
                "        tests_failed += 1\n",
                "\n",
                "print('=' * 70)\n",
                "print('DEEPCFR NETWORK UNIT TESTS')\n",
                "print('=' * 70)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": ".venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Save
with open('test_deepcfr_network.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Created test_deepcfr_network.ipynb with header and imports")
print("Now run: jupyter notebook test_deepcfr_network.ipynb")

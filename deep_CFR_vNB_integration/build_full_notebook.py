import json

# Read test file to extract test functions
with open('test_deepcfr_network.py', 'r') as f:
    test_code = f.read()

# Create full notebook
cells = []

# Header
cells.append({
    "cell_type": "markdown",
    "id": "header",
    "metadata": {},
    "source": [
        "# DeepCFR Network Unit Tests\n",
        "\n",
        "**Comprehensive tests with real MCCFR infosets**\n",
        "\n",
        "Tests all aspects of the DeepCFR network: initialization, forward pass, batch processing, action encoding, edge cases, and consistency."
    ]
})

# Imports cell
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "imports",
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "import os\n",
        "\n",
        "# Add parent directory to path\n",
        "current_dir = os.getcwd()\n",
        "if current_dir.endswith('deep_CFR_vNB_integration'):\n",
        "    parent_dir = os.path.dirname(current_dir)\n",
        "else:\n",
        "    parent_dir = os.path.dirname(os.path.dirname(current_dir))\n",
        "sys.path.insert(0, parent_dir)\n",
        "\n",
        "import torch\n",
        "import random\n",
        "import numpy as np\n",
        "from DeepCFR import DeepCFRModule\n",
        "from infoset_parser import parse_infoset_to_network_input, batch_parse_infosets\n",
        "from mccfr import MCCFR\n",
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
        "        return True\n",
        "    except AssertionError as e:\n",
        "        print(f'✗ {test_name} FAILED: {e}')\n",
        "        tests_failed += 1\n",
        "        return False\n",
        "    except Exception as e:\n",
        "        print(f'✗ {test_name} ERROR: {e}')\n",
        "        tests_failed += 1\n",
        "        return False\n",
        "\n",
        "print('=' * 70)\n",
        "print('DEEPCFR NETWORK UNIT TESTS')\n",
        "print('=' * 70)"
    ]
})

# Extract test sections
test_sections = [
    ("section1", "## 1. Network Initialization Tests", [
        ("test1", "Network creates successfully", """def test_network_creates_successfully():
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=256)
    assert isinstance(network, torch.nn.Module)
    assert network.n_action_history == 20

run_test('Network creates successfully', test_network_creates_successfully)"""),
        
        ("test2", "Network parameter count", """def test_network_parameter_count():
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=256)
    params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    assert params > 0
    print(f'  Network has {params:,} trainable parameters')

run_test('Network has trainable parameters', test_network_parameter_count)"""),
        
        ("test3", "Network embedding layers", """def test_network_embedding_layers():
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=256)
    assert len(network.hand_embeddings) == 3
    assert len(network.board_embeddings) == 5

run_test('Network has correct embeddings', test_network_embedding_layers)""")
    ]),
    
    ("section2", "## 2. Forward Pass - Single Sample Tests", [
        ("test4", "Forward pass preflop", """def test_forward_pass_preflop():
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=256)
    network.eval()
    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)
    with torch.no_grad():
        output = network(cc, ah)
    assert output.shape == torch.Size([1, 9])
    assert output.dtype == torch.float32

run_test('Forward pass: Preflop', test_forward_pass_preflop)"""),
        
        ("test5", "Forward pass with board", """def test_forward_pass_with_board():
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=256)
    network.eval()
    infoset = "S1|H:14s0,13s0,10s1|B:9s0,8s1,7s2,6s0,5s1|A:CRDD"
    cc, ah = parse_infoset_to_network_input(infoset)
    with torch.no_grad():
        output = network(cc, ah)
    assert output.shape == torch.Size([1, 9])
    assert len(cc.canonical_board) == 5

run_test('Forward pass: With board', test_forward_pass_with_board)"""),
        
        ("test6", "Forward pass with action history", """def test_forward_pass_with_action_history():
    network = DeepCFRModule(nhandcards=3, nboardcards=5, n_action_history=20, nresponses=9, dim=256)
    network.eval()
    infoset = "S2|H:14s0,13s0,10s1|B:9s0,8s1,7s2,6s0,5s1|A:CRBXrDFC"
    cc, ah = parse_infoset_to_network_input(infoset)
    with torch.no_grad():
        output = network(cc, ah)
    assert output.shape == torch.Size([1, 9])
    assert ah == "CRBXrDFC"

run_test('Forward pass: With action history', test_forward_pass_with_action_history)""")
    ]),
]

# Add test sections to notebook
for section_id, section_title, tests in test_sections:
    cells.append({"cell_type": "markdown", "id": section_id, "metadata": {}, "source": [section_title]})
    for test_id, test_name, test_code in tests:
        cells.append({"cell_type": "code", "execution_count": None, "id": test_id, "metadata": {}, "outputs": [], "source": test_code})

print(f"Created {len(cells)} cells so far...")

# Save partial
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.5"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('test_deepcfr_network_unit.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Created test_deepcfr_network_unit.ipynb (6 tests added)")
print("  Adding remaining 13 tests...")

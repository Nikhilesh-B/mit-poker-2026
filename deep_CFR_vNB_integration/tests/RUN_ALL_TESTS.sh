#!/bin/bash

# Run all comprehensive tests for Deep CFR integration
# Usage: bash RUN_ALL_TESTS.sh

echo "=================================================="
echo "Deep CFR Integration - Comprehensive Test Suite"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counter
total_passed=0
total_failed=0

# Test 1: Core Components
echo "[1/2] Running Core Components Tests..."
python test_all_units.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Core Components Tests PASSED${NC}"
    total_passed=$((total_passed + 15))
else
    echo -e "${RED}✗ Core Components Tests FAILED${NC}"
    total_failed=$((total_failed + 1))
fi

echo ""
echo ""

# Test 2: DeepCFR Network
echo "[2/2] Running DeepCFR Network Tests..."
python test_deepcfr_network.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ DeepCFR Network Tests PASSED${NC}"
    total_passed=$((total_passed + 19))
else
    echo -e "${RED}✗ DeepCFR Network Tests FAILED${NC}"
    total_failed=$((total_failed + 1))
fi

echo ""
echo "=================================================="
echo "FINAL SUMMARY"
echo "=================================================="

if [ $total_failed -eq 0 ]; then
    echo -e "${GREEN}"
    echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
    echo ""
    echo "Total: 34/34 tests passed"
    echo "Pass Rate: 100.0%"
    echo ""
    echo "🚀 Ready for Step 5: Network-MCCFR Integration"
    echo -e "${NC}"
    exit 0
else
    echo -e "${RED}"
    echo "✗ SOME TESTS FAILED"
    echo ""
    echo "Review failures above before proceeding"
    echo -e "${NC}"
    exit 1
fi

"""
Run ALL comprehensive unit tests in sequence

This script runs all unit tests for the Deep CFR integration project:
1. Infoset Parser tests
2. MCCFR Class tests
3. Action Mapping tests
4. DeepCFR Network tests

Usage: python test_all_comprehensive.py
"""

import sys
import os
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def run_test_file(filename, description):
    """Run a test file and return pass/fail counts"""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Execute the test file
    with open(filename, 'r') as f:
        code = f.read()
    
    # Create a new namespace for execution
    namespace = {}
    
    try:
        exec(code, namespace)
        elapsed = time.time() - start_time
        
        # Extract test results from namespace
        passed = namespace.get('tests_passed', 0)
        failed = namespace.get('tests_failed', 0)
        
        print(f"\nCompleted in {elapsed:.2f}s")
        return passed, failed
        
    except Exception as e:
        print(f"\n✗ ERROR running {filename}: {e}")
        return 0, 1

def main():
    print("=" * 70)
    print("COMPREHENSIVE DEEP CFR INTEGRATION TEST SUITE")
    print("=" * 70)
    print("\nRunning all unit tests...")
    
    total_passed = 0
    total_failed = 0
    
    test_files = [
        ("test_all_units.py", "Core Components (Parser, MCCFR, Action Mapping)"),
        ("test_deepcfr_network.py", "DeepCFR Network"),
    ]
    
    for filename, description in test_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            passed, failed = run_test_file(filepath, description)
            total_passed += passed
            total_failed += failed
        else:
            print(f"\n✗ Test file not found: {filename}")
            total_failed += 1
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    print(f"\nTotal tests passed: {total_passed}")
    print(f"Total tests failed: {total_failed}")
    print(f"Total tests run: {total_passed + total_failed}")
    
    if total_failed == 0:
        print(f"\nPass rate: 100.0%")
        print("\n" + "✓" * 70)
        print("✓✓✓ ALL TESTS PASSED! DEEP CFR INTEGRATION READY! ✓✓✓")
        print("✓" * 70)
        print("\n✅ VERIFIED COMPONENTS:")
        print("   ✓ Infoset Parser")
        print("   ✓ MCCFR Class")
        print("   ✓ Action Mapping")
        print("   ✓ DeepCFR Network")
        print("\n🚀 Ready to proceed to Step 5: Network-MCCFR Integration")
    else:
        pass_rate = 100 * total_passed / (total_passed + total_failed)
        print(f"\nPass rate: {pass_rate:.1f}%")
        print(f"\n✗ {total_failed} TEST(S) FAILED")
        print("Review failures above before proceeding")
    
    print("=" * 70)
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

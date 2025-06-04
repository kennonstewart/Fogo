import unittest
import os
import sys

# Ensure the tests directory is in sys.path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

# Discover and run all tests in the tests package
def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(TESTS_DIR, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

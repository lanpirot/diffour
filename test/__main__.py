# tests/__main__.py

import unittest


def main():
    # Discover and load all test cases from the 'tests' directory
    loader = unittest.TestLoader()
    # Assuming the 'tests' directory is located one level up from the current directory
    tests = loader.discover(start_dir=".", pattern="test_*.py")

    # Run the test suite using TextTestRunner
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(tests)


if __name__ == "__main__":
    main()
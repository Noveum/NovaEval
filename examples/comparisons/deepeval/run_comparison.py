#!/usr/bin/env python3
"""
Main runner script for optimized SDK comparison
"""

import os
import subprocess
import sys
import time


def check_requirements():
    """Check if required packages are installed"""
    try:
        import importlib.util

        # Check if packages are available without importing
        packages = ["datasets", "deepeval", "openai", "novaeval"]
        for package in packages:
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"No module named '{package}'")

        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def check_api_key():
    """Check if OpenAI API key is set"""
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please run: export OPENAI_API_KEY='your_key_here'")
        return False
    print("✅ OpenAI API key is set")
    return True


def run_novaeval():
    """Run NovaEval evaluation"""
    print("\n🔵 Running NovaEval evaluation...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "scripts/novaeval_evaluator_fixed.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        end_time = time.time()

        if result.returncode == 0:
            print(f"✅ NovaEval completed in {end_time - start_time:.2f}s")
            return True
        else:
            print(f"❌ NovaEval failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ NovaEval error: {e}")
        return False


def run_deepeval():
    """Run DeepEval evaluation"""
    print("\n🟢 Running DeepEval evaluation...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "scripts/deepeval_final_comparison.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        end_time = time.time()

        if result.returncode == 0:
            print(f"✅ DeepEval completed in {end_time - start_time:.2f}s")
            return True
        else:
            print(f"❌ DeepEval failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ DeepEval error: {e}")
        return False


def main():
    """Main execution function"""
    print("🚀 Optimized SDK Comparison Runner")
    print("=" * 50)

    # Check prerequisites
    if not check_requirements():
        sys.exit(1)

    if not check_api_key():
        sys.exit(1)

    print("\n📋 Starting evaluations...")

    # Run evaluations
    nova_success = run_novaeval()
    deep_success = run_deepeval()

    # Summary
    print("\n📊 Evaluation Summary:")
    print("=" * 30)
    print(f"NovaEval: {'✅ Success' if nova_success else '❌ Failed'}")
    print(f"DeepEval: {'✅ Success' if deep_success else '❌ Failed'}")

    if nova_success and deep_success:
        print("\n🎉 Both evaluations completed successfully!")
        print("📁 Check the results/ directory for detailed outputs")
        print("📖 See docs/final_optimized_comparison_report.md for analysis")
    else:
        print("\n⚠️  Some evaluations failed. Check error messages above.")


if __name__ == "__main__":
    main()

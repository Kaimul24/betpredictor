#!/usr/bin/env python3
"""
Test runner script for BetPredictor loaders.

This script provides convenient ways to run different subsets of tests
and generate reports.
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run BetPredictor loader tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--game", action="store_true", help="Run GameLoader tests only")
    parser.add_argument("--odds", action="store_true", help="Run OddsLoader tests only")
    parser.add_argument("--player", action="store_true", help="Run PlayerLoader tests only")
    parser.add_argument("--team", action="store_true", help="Run TeamLoader tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    success = True
    
    if args.all or not any([args.game, args.odds, args.player, args.team]):
        # Run all tests
        cmd = base_cmd + ["tests/loaders/"]
        if args.parallel:
            cmd.extend(["-n", "auto"])
        if args.coverage:
            cmd.extend(["--cov=data.loaders", "--cov-report=html", "--cov-report=term"])
        
        success &= run_command(cmd, "All loader tests")
    
    else:
        if args.game:
            cmd = base_cmd + ["tests/loaders/test_game_loader.py"]
            success &= run_command(cmd, "GameLoader tests")
        
        if args.odds:
            cmd = base_cmd + ["tests/loaders/test_odds_loader.py"]
            success &= run_command(cmd, "OddsLoader tests")
        
        if args.player:
            cmd = base_cmd + ["tests/loaders/test_player_loader.py"]
            success &= run_command(cmd, "PlayerLoader tests")
        
        if args.team:
            cmd = base_cmd + ["tests/loaders/test_team_loader.py"]
            success &= run_command(cmd, "TeamLoader tests")
    
    print(f"\n{'='*60}")
    if success:
        print("All requested tests completed successfully!")
    else:
        print("Some tests failed. Check output above for details.")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

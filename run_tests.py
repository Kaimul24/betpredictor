#!/usr/bin/env python3
"""
Test runner script for BetPredictor.

This script provides convenient ways to run different subsets of tests
and generate reports for loaders and features.
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
    parser = argparse.ArgumentParser(description="Run BetPredictor tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Loader tests
    parser.add_argument("--loaders", action="store_true", help="Run all loader tests")
    parser.add_argument("--game", action="store_true", help="Run GameLoader tests only")
    parser.add_argument("--odds", action="store_true", help="Run OddsLoader tests only")
    parser.add_argument("--player", action="store_true", help="Run PlayerLoader tests only")
    parser.add_argument("--team", action="store_true", help="Run TeamLoader tests only")
    
    # Feature tests  
    parser.add_argument("--features", action="store_true", help="Run all feature tests")
    parser.add_argument("--batting", action="store_true", help="Run batting features tests only")
    parser.add_argument("--pitching", action="store_true", help="Run pitching features tests only")
    parser.add_argument("--context", action="store_true", help="Run GameContextFeatures tests only")
    parser.add_argument("--team-features", action="store_true", help="Run TeamFeatures tests only")
    parser.add_argument("--pipeline", action="store_true", help="Run FeaturePipeline tests only")
    
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    base_cmd = [sys.executable, "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    success = True
    
    if args.all or not any([args.loaders, args.features, args.game, args.odds, args.player, args.team, args.batting, args.pitching, args.context, args.team_features, args.pipeline]):
        # Run all tests
        cmd = base_cmd + ["tests/"]
        if args.parallel:
            cmd.extend(["-n", "auto"])
        if args.coverage:
            cmd.extend(["--cov=data", "--cov-report=html", "--cov-report=term"])
        
        success &= run_command(cmd, "All tests")
    
    else:
        # Loader tests
        if args.loaders:
            cmd = base_cmd + ["tests/loaders/"]
            if args.parallel:
                cmd.extend(["-n", "auto"])
            if args.coverage:
                cmd.extend(["--cov=data.loaders", "--cov-report=html", "--cov-report=term"])
            success &= run_command(cmd, "All loader tests")
        
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
        
        # Feature tests
        if args.features:
            cmd = base_cmd + ["tests/features/"]
            if args.parallel:
                cmd.extend(["-n", "auto"])
            if args.coverage:
                cmd.extend(["--cov=data.features", "--cov-report=html", "--cov-report=term"])
            success &= run_command(cmd, "All feature tests")
        
        if args.batting:
            cmd = base_cmd + ["tests/features/test_batting_features.py"]
            success &= run_command(cmd, "batting features tests")

        if args.pitching:
            cmd = base_cmd + ["tests/features/test_pitching_features.py"]
            success &= run_command(cmd, "pitching features tests")
        
        if args.context:
            cmd = base_cmd + ["tests/features/test_context_features.py"]
            success &= run_command(cmd, "GameContextFeatures tests")
        
        if args.team_features:
            cmd = base_cmd + ["tests/features/test_team_features.py"]
            success &= run_command(cmd, "TeamFeatures tests")
        
        if args.pipeline:
            cmd = base_cmd + ["tests/features/test_feature_pipeline.py"]
            success &= run_command(cmd, "FeaturePipeline tests")
    
    print(f"\n{'='*60}")
    if success:
        print("All requested tests completed successfully!")
    else:
        print("Some tests failed. Check output above for details.")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

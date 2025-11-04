#!/usr/bin/env python3
"""
AP AI Classifier - Main Evaluation Script

This script runs evaluations for all three approaches:
1. ML Baseline (TF-IDF)
2. ML with OpenAI Embeddings
3. LLM with Few-Shot Learning (model from OPENAI_MODEL env var)

Usage:
    python evaluate.py [--experiment {all,ml-tfidf-baseline,ml-openai,llm}]
    
Environment:
    Set OPENAI_API_KEY in .env file or as environment variable
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from getpass import getpass

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import config for dynamic model names
from ap_ai_classifier.config import LLM_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for user-facing output
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI client libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def check_openai_api_key():
    """Check for OpenAI API key in environment or .env file."""
    # Check if already in environment
    if os.getenv("OPENAI_API_KEY"):
        logger.info("✅ OpenAI API key found in environment")
        return True
    
    # Check .env file
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        os.environ["OPENAI_API_KEY"] = key
                        logger.info("✅ OpenAI API key loaded from .env file")
                        return True
    
    # Prompt user for key
    logger.warning("\n⚠️  OpenAI API key not found!")
    logger.info("\nYou have two options:")
    logger.info("1. Create a .env file with: OPENAI_API_KEY=your-key-here")
    logger.info("2. Enter your API key now (will be saved to .env)")
    
    choice = input("\nEnter your OpenAI API key now? (y/n): ").strip().lower()
    
    if choice == 'y':
        api_key = getpass("Enter your OpenAI API key: ").strip()
        
        if api_key:
            # Save to .env
            with open(env_file, "a") as f:
                f.write(f"\nOPENAI_API_KEY={api_key}\n")
            
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info("✅ API key saved to .env file")
            return True
        else:
            logger.error("❌ No API key provided")
            return False
    else:
        logger.info("\n📝 Please create a .env file with your API key and run again:")
        logger.info(f"   echo 'OPENAI_API_KEY=your-key-here' > {env_file}")
        return False


def check_openai_model():
    """Check for OpenAI model in environment or .env file, prompt if missing."""
    env_file = PROJECT_ROOT / ".env"
    
    # Check if already in environment
    if os.getenv("OPENAI_MODEL"):
        model = os.getenv("OPENAI_MODEL")
        logger.info(f"✅ OpenAI model found in environment: {model}")
        return True
    
    # Check .env file
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip().startswith("OPENAI_MODEL="):
                    model = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if model:
                        os.environ["OPENAI_MODEL"] = model
                        logger.info(f"✅ OpenAI model loaded from .env file: {model}")
                        return True
    
    # Prompt user for model
    logger.warning("\n⚠️  OpenAI model (OPENAI_MODEL) not found!")
    logger.info("\nPlease choose an LLM model to use for experiments:")
    logger.info("1. gpt-4.1-mini (recommended - good balance of performance and cost)")
    logger.info("2. gpt-5-mini")
    logger.info("3. Cancel and add OPENAI_MODEL to .env file manually")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        model = "gpt-4.1-mini"
    elif choice == "2":
        model = "gpt-5-mini"
    elif choice == "3":
        logger.info("\n📝 Please add OPENAI_MODEL to your .env file:")
        logger.info(f"   echo 'OPENAI_MODEL=gpt-4.1-mini' >> {env_file}")
        logger.info("   or")
        logger.info(f"   echo 'OPENAI_MODEL=gpt-5-mini' >> {env_file}")
        return False
    else:
        logger.error("❌ Invalid choice")
        return False
    
    # Save to .env
    if model:
        # Read existing .env content
        env_content = ""
        if env_file.exists():
            with open(env_file, "r") as f:
                env_content = f.read()
        
        # Check if OPENAI_MODEL already exists (shouldn't, but just in case)
        if "OPENAI_MODEL=" not in env_content:
            with open(env_file, "a") as f:
                f.write(f"\nOPENAI_MODEL={model}\n")
        
        os.environ["OPENAI_MODEL"] = model
        logger.info(f"✅ Model '{model}' saved to .env file")
        
        return True
    
    return False


def run_ml_baseline(tune_hyperparameters=False):
    """Run ML baseline evaluation (TF-IDF)."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1: ML Baseline (TF-IDF)")
    logger.info("="*80 + "\n")
    
    import sys
    import importlib.util
    
    # Load the evaluator script
    spec = importlib.util.spec_from_file_location(
        "ml_baseline",
        PROJECT_ROOT / "src/ap_ai_classifier/evaluators/01_ml_tfidf_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ml_baseline"] = module
    spec.loader.exec_module(module)
    
    result = module.main(tune_hyperparameters=tune_hyperparameters)
    return result
    

def run_ml_openai(tune_hyperparameters=False):
    """Run ML with OpenAI embeddings evaluation."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2: ML with OpenAI Embeddings")
    logger.info("="*80 + "\n")
    
    import sys
    import importlib.util
    
    # Load the evaluator script
    spec = importlib.util.spec_from_file_location(
        "ml_openai",
        PROJECT_ROOT / "src/ap_ai_classifier/evaluators/02_ml_openai_embeddings.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ml_openai"] = module
    spec.loader.exec_module(module)
    
    logger.warning("⚠️  This will make ~2500 OpenAI embedding API calls")
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.warning("❌ Skipped")
        return None
    
    result = module.main(tune_hyperparameters=tune_hyperparameters)
    return result


def run_llm_evaluation():
    """Run LLM evaluation (model from OPENAI_MODEL env var with few-shot learning)."""
    # Get the current model from config (may have been updated)
    from ap_ai_classifier.config import LLM_MODEL as current_model
    logger.info("\n" + "="*80)
    logger.info(f"EXPERIMENT 3: LLM with Few-Shot Learning ({current_model})")
    logger.info("="*80 + "\n")
    
    import sys
    import importlib.util
    
    # Load the evaluator script
    spec = importlib.util.spec_from_file_location(
        "llm_final",
        PROJECT_ROOT / "src/ap_ai_classifier/evaluators/03_llm_final.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_final"] = module
    spec.loader.exec_module(module)
    
    logger.warning("⚠️  This will make ~500 LLM API calls + embedding calls")
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.warning("❌ Skipped")
        return None
    
    result = module.main()
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AP AI Classifier - Run evaluations for ML and LLM approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                    # Run all experiments (with prompts)
  python evaluate.py --experiment all   # Run all experiments
  python evaluate.py --experiment ml-tfidf-baseline
  python evaluate.py --experiment ml-openai
  python evaluate.py --experiment llm

Results:
  All results are saved to docs/results/ directory
        """
    )
    
    parser.add_argument(
        "--experiment",
        choices=["all", "ml-tfidf-baseline", "ml-openai", "llm"],
        default="all",
        help="Which experiment to run (default: all)"
    )
    
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning (default: False)"
    )
    
    args = parser.parse_args()
    
    # Determine if hyperparameter tuning should be enabled
    # If flag not set, prompt user for experiments that support tuning
    tune_hyperparameters = args.tune_hyperparameters
    
    if not tune_hyperparameters and args.experiment in ["all", "ml-tfidf-baseline", "ml-openai"]:
        logger.info("\n🔍 Hyperparameter Tuning:")
        logger.info("   This will significantly increase runtime but may improve performance.")
        tune_choice = input("   Enable hyperparameter tuning? (y/n): ").strip().lower()
        tune_hyperparameters = (tune_choice == 'y')
    
    if tune_hyperparameters:
        logger.info(f"\n✅ Hyperparameter tuning enabled")
    else:
        logger.info(f"\n⚙️  Using default hyperparameters")
    
    logger.info("\n" + "="*80)
    logger.info("AP AI CLASSIFIER - EVALUATION SUITE")
    logger.info("="*80)
    logger.info("\nInvoice Line Item Classification")
    logger.info("Target: F1-Macro ≥ 0.70 across NOMINAL, DEPARTMENT, and TC fields")
    logger.info("="*80 + "\n")
    
    # Store results from all experiments
    all_results = []
    
    # Experiment 1: ML Baseline (no API key needed)
    if args.experiment in ["all", "ml-tfidf-baseline"]:
        try:
            result = run_ml_baseline(tune_hyperparameters=tune_hyperparameters)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ ML Baseline failed: {e}")
            if args.experiment == "ml-tfidf-baseline":
                sys.exit(1)
    
    # Experiments 2 & 3 need OpenAI API key
    if args.experiment in ["all", "ml-openai", "llm"]:
        if not check_openai_api_key():
            logger.error("\n❌ Cannot run OpenAI experiments without API key")
            sys.exit(1)
    
    # Experiment 3 (LLM) also needs OPENAI_MODEL
    if args.experiment in ["all", "llm"]:
        if not check_openai_model():
            logger.error("\n❌ Cannot run LLM experiment without OPENAI_MODEL")
            sys.exit(1)
        # Reload config module to pick up the updated model from environment
        import importlib
        import ap_ai_classifier.config
        importlib.reload(ap_ai_classifier.config)
    
    # Experiment 2: ML + OpenAI
    if args.experiment in ["all", "ml-openai"]:
        try:
            result = run_ml_openai(tune_hyperparameters=tune_hyperparameters)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ ML + OpenAI failed: {e}")
            if args.experiment == "ml-openai":
                sys.exit(1)
    
    # Experiment 3: LLM
    if args.experiment in ["all", "llm"]:
        try:
            result = run_llm_evaluation()
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ LLM Evaluation failed: {e}")
            if args.experiment == "llm":
                sys.exit(1)
    
    # Summary with dynamic winner determination
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    
    # Show results summary
    if all_results:
        num_experiments = len(all_results)
        
        # Helper function to get comparison F1 score
        def get_comparison_f1(result):
            if 'overall_f1_macro' in result:
                return result['overall_f1_macro']
            elif 'avg_f1_macro' in result:
                return result['avg_f1_macro']
            elif 'best_f1_macro' in result:
                return result['best_f1_macro']
            return 0.0
        
        if num_experiments == 1:
            # Single experiment - just show its results with explicit method/model name
            result = all_results[0]
            f1_score = get_comparison_f1(result)
            experiment_name = result['experiment']
            
            # Get specific method/model name
            method_name = None
            if 'best_method' in result:
                method_name = result['best_method']
            elif 'experiment' in result and ('(' in result['experiment']):
                # Extract model name from experiment string (e.g., "LLM with Few-Shot Learning")
                method_name = result['experiment'].split('(')[1].rstrip(')') if '(' in result['experiment'] else None
            
            if method_name:
                logger.info(f"\n📊 {experiment_name} Results:")
                logger.info(f"   Best Method: {method_name}")
                logger.info(f"   Average F1-Macro (across all fields): {f1_score:.4f}")
            else:
                logger.info(f"\n📊 {experiment_name} Results:")
                logger.info(f"   Average F1-Macro (across all fields): {f1_score:.4f}")
            
            if f1_score >= 0.70:
                logger.info(f"   ✅ Exceeds target by {(f1_score - 0.70)*100:.1f}%")
            else:
                logger.warning(f"   ⚠️  Gap to target: {(0.70 - f1_score)*100:.1f}%")
        
        else:
            # Multiple experiments - show comparison and winner with explicit method/model names
            logger.info(f"\n📊 Comparison of {num_experiments} Experiments:")
            logger.info("-" * 80)
            
            for result in all_results:
                f1_score = get_comparison_f1(result)
                experiment_name = result['experiment']
                
                # Get specific method/model name
                method_name = None
                if 'best_method' in result:
                    method_name = result['best_method']
                elif 'experiment' in result and '(' in result['experiment']:
                    # Extract model name from experiment string (e.g., "LLM with Few-Shot Learning")
                    method_name = result['experiment'].split('(')[1].rstrip(')')
                
                if method_name:
                    display_name = f"{experiment_name} ({method_name})"
                else:
                    display_name = experiment_name
                
                marker = " ✅" if f1_score >= 0.70 else ""
                logger.info(f"   {display_name:<55} Avg F1-Macro: {f1_score:.4f}{marker}")
            
            # Determine winner
            winner = max(all_results, key=get_comparison_f1)
            winner_f1 = get_comparison_f1(winner)
            winner_name = winner['experiment']
            
            # Get winner's specific method/model name
            winner_method = None
            if 'best_method' in winner:
                winner_method = winner['best_method']
            elif 'experiment' in winner and '(' in winner['experiment']:
                # Extract model name from experiment string (e.g., "LLM with Few-Shot Learning")
                winner_method = winner['experiment'].split('(')[1].rstrip(')')
            
            logger.info("-" * 80)
            if winner_method:
                logger.info(f"\n🏆 Winner: {winner_name}")
                logger.info(f"   Method: {winner_method}")
                logger.info(f"   Average F1-Macro (across all fields): {winner_f1:.4f}")
            else:
                logger.info(f"\n🏆 Winner: {winner_name}")
                logger.info(f"   Average F1-Macro (across all fields): {winner_f1:.4f}")
            
            if winner_f1 >= 0.70:
                logger.info(f"   ✅ Exceeds target by {(winner_f1 - 0.70)*100:.1f}%")
            else:
                logger.warning(f"   ⚠️  Gap to target: {(0.70 - winner_f1)*100:.1f}%")
    else:
        logger.warning("\n⚠️  No results available")
    
    logger.info("\n📊 Results saved to: docs/results/")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Training script for XAUUSD Scalping System."""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import get_config
from data import DataLoader
from features import FeatureEngineer, generate_training_data
from models.trainer import ModelTrainer
from models.ensemble import EnsembleModel
from models.persistence import ModelPersistence
from logging import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train XAUUSD Scalping Models")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--symbol", type=str, default="XAUUSD=X", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1m", help="Data interval")
    parser.add_argument("--period", type=str, default="30d", help="Data period")
    parser.add_argument("--model", type=str, default="ensemble", help="Model name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--target-horizon", type=int, default=5, help="Target horizon (minutes)")
    parser.add_argument("--threshold-long", type=float, default=0.0015, help="Long threshold")
    parser.add_argument("--threshold-short", type=float, default=-0.0015, help="Short threshold")
    parser.add_argument("--labeling", type=str, default="triple_barrier", help="Labeling method")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--save-model", action="store_true", help="Save model after training")
    parser.add_argument("--export-production", action="store_true", help="Export for production")
    
    args = parser.parse_args()
    
    # Setup logging
    config_path = args.config
    setup_logging(config_path=config_path, log_level="INFO")
    logger = get_logger("train")
    
    logger.info("Starting XAUUSD Scalping Model Training")
    logger.info(f"Config: {config_path}")
    logger.info(f"Symbol: {args.symbol}, Interval: {args.interval}, Period: {args.period}")
    
    # Load configuration
    config = get_config(config_path)
    
    # Load data
    logger.info("Loading market data...")
    loader = DataLoader(config_path)
    df = loader.load_ohlcv(interval=args.interval, period=args.period)
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Validate data quality
    quality = loader.validate_data_quality(df)
    logger.info(f"Data quality: {quality}")
    
    if not quality['valid']:
        logger.warning(f"Data quality issues: {quality['issues']}")
    
    # Generate features
    logger.info("Generating features...")
    engineer = FeatureEngineer(config_path)
    features = engineer.fit_transform(df)
    
    if features.empty:
        logger.error("Feature generation failed. Exiting.")
        return 1
    
    logger.info(f"Generated {len(features.columns)} features")
    
    # Generate targets
    logger.info("Generating targets...")
    target_config = config.get_section('target')
    features, target = generate_training_data(
        df,
        target_horizon=target_config.get('horizon', args.target_horizon),
        threshold_long=target_config.get('threshold_long', args.threshold_long),
        threshold_short=target_config.get('threshold_short', args.threshold_short),
        method=target_config.get('labeling_method', args.labeling),
    )
    
    logger.info(f"Target distribution: {target.value_counts().to_dict()}")
    
    # Align features and target
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    
    logger.info(f"Training data: {len(features)} samples, {len(features.columns)} features")
    
    # Train models
    logger.info("Training models...")
    trainer = ModelTrainer(config_path)
    
    results = trainer.train_all(
        features,
        target,
        test_size=args.test_size,
        validation_size=args.val_size,
    )
    
    # Cross-validation
    logger.info("Running cross-validation...")
    cv_results = trainer.cross_validate(features, target, cv_folds=args.cv_folds)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate_on_test()
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"\n{name}:")
            print(f"  Validation F1: {metrics.get('f1_macro', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision_macro', 0):.4f}")
            print(f"  Recall: {metrics.get('recall_macro', 0):.4f}")
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    for name, cv in cv_results.items():
        if 'error' not in cv:
            print(f"\n{name}:")
            print(f"  Mean F1: {cv['mean']:.4f} (+/- {cv['std']:.4f})")
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    for name, metrics in test_results.items():
        if 'error' not in metrics:
            print(f"\n{name}:")
            print(f"  Test F1: {metrics.get('f1_macro', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    # Get best model
    best_model = trainer.best_model_name
    best_score = trainer.best_score
    
    print(f"\nBest Model: {best_model} (F1: {best_score:.4f})")
    
    # Feature importance
    importance = trainer.get_feature_importance(best_model, top_n=20)
    print("\nTop 20 Features:")
    print(importance.to_string(index=False))
    
    # Save model
    if args.save_model or args.export_production:
        logger.info("Saving model...")
        persistence = ModelPersistence()
        
        version = persistence.save(
            model=trainer.models[best_model],
            model_name=args.model,
            scaler=trainer.scalers.get(best_model),
            feature_names=trainer.feature_names,
            metrics=test_results.get(best_model, {}),
            feature_importance=importance,
            config=config.get_section('models'),
        )
        
        logger.info(f"Model saved to {version}")
        
        if args.export_production:
            prod_path = persistence.export_for_production(args.model, "latest")
            logger.info(f"Exported production model to {prod_path}")
    
    # Save feature names
    feature_path = Path("models") / args.model / "latest" / "features.json"
    if feature_path.exists():
        logger.info(f"Feature names saved to {feature_path}")
    
    logger.info("Training completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

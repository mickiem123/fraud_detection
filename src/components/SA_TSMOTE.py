import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict

# ==============================================================================
# 1. CONFIGURATION
# Holds all parameters for the data generation process.
# ==============================================================================

@dataclass
class GenerationConfig:
    """Configuration for the synthetic data generation pipeline."""
    target_fraud_ratio: float = 0.20  # We want the final dataset to have 20% fraud
    
    # Tier 1: Ground Truth Budget Allocation
    gt_budget_alloc: float = 0.60 # 60% of total synthetic samples
    gt_scenario1_alloc: float = 0.15 # of gt_budget
    gt_scenario2_alloc: float = 0.45 # of gt_budget
    gt_scenario3_alloc: float = 0.40 # of gt_budget
    
    # Tier 2: Robustness Budget Allocation
    robustness_budget_alloc: float = 0.20 # 20% of total synthetic samples

    # Tier 3: Boundary Budget Allocation
    boundary_budget_alloc: float = 0.20 # 20% of total synthetic samples
    
    # Names of raw features to be used in generation and LSTM
    raw_feature_names: List[str] = field(default_factory=lambda: ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT'])


# ==============================================================================
# 2. ABSTRACT BASE CLASS FOR A PIPELINE STEP
# Defines the "contract" for all steps.
# ==============================================================================

class PipelineStep(ABC):
    """Abstract base class for a single step in the data generation pipeline."""
    @abstractmethod
    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        """
        Processes a DataFrame and returns a transformed DataFrame.

        Args:
            df (pd.DataFrame): The data from the previous step.
            context (Dict): A dictionary for sharing objects (like a fitted scaler
                            or budget info) between steps.
        
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

# ==============================================================================
# 3. PIPELINE ORCHESTRATOR CLASS
# Manages the sequence of steps and the flow of data.
# ==============================================================================

class DataGenerationPipeline:
    """Orchestrates the execution of a sequence of data generation and processing steps."""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, initial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the entire pipeline sequentially.

        Args:
            initial_df (pd.DataFrame): The starting, original training DataFrame.
        
        Returns:
            pd.DataFrame: The final, fully augmented DataFrame.
        """
        df = initial_df.copy()
        # The context dictionary will be passed through and modified by each step
        context = {} 

        print("--- Starting Data Generation Pipeline ---")
        for step in self.steps:
            step_name = step.__class__.__name__
            print(f"\nExecuting Step: {step_name}")
            df = step.process(df, context)
            print(f"  -> DataFrame shape after {step_name}: {df.shape}")
        print("\n--- Pipeline Finished ---")
        
        return df, context



# --- PHASE 1: Budgeting and Ground Truth Raw Generation ---

class BudgetingStep(PipelineStep):
    """Calculates the number of samples to generate for each tier and scenario."""
    def __init__(self, config: GenerationConfig):
        self.config = config

    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        # This step doesn't modify the df, only populates the context dict.
        # ... logic to calculate budgets for s1, s2, s3, robustness, boundary ...
        context['budget'] = { 's1': 100, 's2': 450, 's3': 400, 'robustness': 200, 'boundary': 200 } # Example
        print(f"  -> Budget calculated and stored in context: {context['budget']}")
        return df

class GroundTruthGeneratorStep(PipelineStep):
    """Generates raw synthetic transactions for Scenarios 1, 2, and 3."""
    def __init__(self, config: GenerationConfig):
        self.config = config

    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        budget = context.get('budget', {})
        s1_budget = budget.get('s1', 0)
        s2_budget = budget.get('s2', 0)
        s3_budget = budget.get('s3', 0)
        print(f"  -> Generating {s1_budget+s2_budget+s3_budget} raw ground truth samples...")
        # ... TO-DO: Implement generation logic for S1, S2, S3 raw events ...
        # ... and concat them with the input df ...
        return df

# --- PHASE 2: Foundational Processing ---

class FeatureEngineeringStep(PipelineStep):
    """Runs the full feature engineering pipeline on the augmented raw data."""
    def __init__(self, preprocessor_pipeline_class):
        self.preprocessor_class = preprocessor_pipeline_class

    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        print("  -> Re-computing all aggregated features...")
        # engineered_df = self.preprocessor_class(df, add_method=...).process()
        # return engineered_df
        return df # Placeholder

class ScalingStep(PipelineStep):
    """Fits a scaler on original normal data and transforms the entire dataset."""
    def __init__(self, original_df: pd.DataFrame):
        self.original_df_normal_only = original_df[original_df['TX_FRAUD'] == 0]
    
    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        print("  -> Fitting scaler and transforming data...")
        # ... TO-DO: Fit a scaler on self.original_df_normal_only ...
        # ... TO-DO: Transform df ...
        # ... TO-DO: Store the fitted scaler in context -> context['scaler'] = fitted_scaler ...
        return df

# --- PHASE 3: Advanced Generation in Scaled Space ---

class RobustnessGeneratorStep(PipelineStep):
    """Generates samples by perturbing existing scaled fraud vectors."""
    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        budget = context.get('budget', {}).get('robustness', 0)
        print(f"  -> Generating {budget} robustness samples...")
        # ... TO-DO: Implement perturbation logic on scaled features ...
        return df

class BoundaryGeneratorStep(PipelineStep):
    """Generates samples via hard negative mining."""
    def __init__(self, baseline_model):
        self.baseline_model = baseline_model
    
    def process(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        budget = context.get('budget', {}).get('boundary', 0)
        print(f"  -> Generating {budget} boundary samples...")
        # ... TO-DO: Use self.baseline_model to find hard negatives and generate samples ...
        return df
"""
Run Baseline Training Experiment on SageMaker Pipeline
Execute the pipeline with baseline hyperparameter configuration
"""
import boto3
from datetime import datetime

PIPELINE_NAME = 'FlightDelayTrainingPipeline'

# Baseline hyperparameters
BASELINE_PARAMETERS = {
    'MaxDepth': 6,
    'Eta': 0.1,
    'NumRound': 100,
    'Subsample': 0.8,
    'ColsampleByTree': 0.8
}

def check_pipeline_exists():
    """Verify pipeline exists before running"""
    sm_client = boto3.client('sagemaker')
    
    try:
        response = sm_client.describe_pipeline(PipelineName=PIPELINE_NAME)
        print(f"✅ Found pipeline: {PIPELINE_NAME}")
        return True
    except sm_client.exceptions.ResourceNotFound:
        print(f"❌ Pipeline '{PIPELINE_NAME}' not found!")
        print(f"   Run 'python pipeline_definition.py' first to create the pipeline.")
        return False

def run_baseline_experiment():
    """Run baseline training experiment"""
    sm_client = boto3.client('sagemaker')
    
    experiment_name = f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"\n{'='*70}")
    print(f"🚀 STARTING BASELINE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Experiment Name: {experiment_name}")
    print(f"\nHyperparameters:")
    for key, value in BASELINE_PARAMETERS.items():
        print(f"   {key}: {value}")
    print(f"{'='*70}\n")
    
    # Start pipeline execution
    response = sm_client.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=experiment_name,
        PipelineParameters=[
            {'Name': key, 'Value': str(value)}
            for key, value in BASELINE_PARAMETERS.items()
        ]
    )
    
    execution_arn = response['PipelineExecutionArn']
    
    print(f"✅ Experiment started!")
    print(f"\n📋 Execution Details:")
    print(f"   ARN: {execution_arn}")
    print(f"   Name: {experiment_name}")
    
    print(f"\n⏱️  Expected Duration:")
    print(f"   Training: ~10-15 minutes")
    print(f"   Evaluation: ~2-5 minutes")
    print(f"   Registration: ~1-2 minutes (if F1 >= 0.70)")
    print(f"   Total: ~15-25 minutes")
    
    print(f"\n📊 Monitor Progress:")
    print(f"   1. Open SageMaker Studio")
    print(f"   2. Navigate to: Pipelines → {PIPELINE_NAME}")
    print(f"   3. Find execution: {experiment_name}")
    print(f"   4. Watch pipeline steps execute")
    
    print(f"\n📦 Expected Outcome:")
    print(f"   If F1 >= 0.70:")
    print(f"      → Model registered in Model Registry: 'flight-delay-models'")
    print(f"      → Status: PendingManualApproval")
    print(f"   If F1 < 0.70:")
    print(f"      → Pipeline stops at condition check")
    print(f"      → Model NOT registered")
    
    return execution_arn, experiment_name

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"FLIGHT DELAY PREDICTION - BASELINE EXPERIMENT")
    print(f"{'='*70}\n")
    
    # Check pipeline exists
    if not check_pipeline_exists():
        exit(1)
    
    # Run experiment
    execution_arn, experiment_name = run_baseline_experiment()
    
    print(f"\n{'='*70}")
    print(f"🎉 EXPERIMENT SUBMITTED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nNext Steps:")
    print(f"   1. Monitor in SageMaker Studio (see instructions above)")
    print(f"   2. Wait for completion (~15-25 minutes)")
    print(f"   3. Check Model Registry for registered model")
    print(f"\n💡 Tip: You can check execution status with:")
    print(f"   aws sagemaker describe-pipeline-execution \\")
    print(f"       --pipeline-execution-arn {execution_arn}")
    print(f"{'='*70}\n")
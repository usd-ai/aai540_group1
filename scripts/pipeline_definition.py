"""
SageMaker Pipeline Definition for Flight Delay Prediction
Creates an automated ML pipeline with training, evaluation, and model registration
"""
import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile

# ===========================
# CONFIGURATION
# ===========================
BUCKET = 'sagemaker-us-east-1-425709451100'
PREFIX = 'aai540-group1'
MODEL_PACKAGE_GROUP = 'flight-delay-models'
PIPELINE_NAME = 'FlightDelayTrainingPipeline'

# Paths
TRAIN_DATA_PATH = f's3://{BUCKET}/{PREFIX}/data/train/'
VAL_DATA_PATH = f's3://{BUCKET}/{PREFIX}/data/validation/'
TEST_DATA_PATH = f's3://{BUCKET}/{PREFIX}/data/test/'
MODEL_OUTPUT_PATH = f's3://{BUCKET}/{PREFIX}/models/'
EVALUATION_OUTPUT_PATH = f's3://{BUCKET}/{PREFIX}/evaluation/'
SCRIPT_PATH = f's3://{BUCKET}/{PREFIX}/scripts/evaluate.py'

# ===========================
# UPLOAD EVALUATION SCRIPT TO S3
# ===========================
def upload_evaluation_script():
    """Upload evaluation script to S3"""
    print("📤 Uploading evaluation script to S3...")
    
    s3 = boto3.client('s3')
    local_script = 'scripts/evaluate.py'
    s3_key = f'{PREFIX}/scripts/evaluate.py'
    
    if not os.path.exists(local_script):
        raise FileNotFoundError(f"Evaluation script not found at {local_script}")
    
    s3.upload_file(local_script, BUCKET, s3_key)
    print(f"✅ Uploaded to s3://{BUCKET}/{s3_key}")

# ===========================
# DEFINE PIPELINE
# ===========================
def create_pipeline():
    """Create SageMaker Pipeline"""
    
    # Setup
    role = get_execution_role()
    session = sagemaker.Session()
    region = session.boto_region_name
    
    print(f"\n{'='*70}")
    print(f"CREATING SAGEMAKER PIPELINE: {PIPELINE_NAME}")
    print(f"{'='*70}")
    print(f"Region: {region}")
    print(f"Bucket: {BUCKET}")
    print(f"Prefix: {PREFIX}")
    
    # ===========================
    # PIPELINE PARAMETERS
    # ===========================
    max_depth = ParameterInteger(name='MaxDepth', default_value=6)
    eta = ParameterFloat(name='Eta', default_value=0.1)
    num_round = ParameterInteger(name='NumRound', default_value=100)
    subsample = ParameterFloat(name='Subsample', default_value=0.8)
    colsample_bytree = ParameterFloat(name='ColsampleByTree', default_value=0.8)
    scale_pos_weight = ParameterFloat(name='ScalePosWeight', default_value=5.5) 
    min_child_weight = ParameterInteger(name='MinChildWeight', default_value=1)
    
    print(f"\n📋 Pipeline Parameters (defaults):")
    print(f"   MaxDepth: {max_depth.default_value}")
    print(f"   Eta: {eta.default_value}")
    print(f"   NumRound: {num_round.default_value}")
    print(f"   Subsample: {subsample.default_value}")
    print(f"   ColsampleByTree: {colsample_bytree.default_value}")
    print(f"   ScalePosWeight: {scale_pos_weight.default_value}")
    print(f"   MinChildWeight: {min_child_weight.default_value}")
    
    
    # ===========================
    # STEP 1: TRAINING
    # ===========================
    print(f"\n🔧 Defining Training Step...")
    
    xgboost_container = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.5-1'
    )
    
    xgb_estimator = Estimator(
        image_uri=xgboost_container,
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=MODEL_OUTPUT_PATH,
        sagemaker_session=session,
        base_job_name='flight-delay-training'
    )
    
    xgb_estimator.set_hyperparameters(
        objective='binary:logistic',
        max_depth=max_depth,
        eta=eta,
        num_round=num_round,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,        
        eval_metric='auc'
    )
    
    training_step = TrainingStep(
        name='TrainFlightDelayModel',
        estimator=xgb_estimator,
        inputs={
            'train': TrainingInput(s3_data=TRAIN_DATA_PATH, content_type='text/csv'),
            'validation': TrainingInput(s3_data=VAL_DATA_PATH, content_type='text/csv')
        }
    )
    
    print("   ✅ Training step defined")
    
    # ===========================
    # STEP 2: EVALUATION
    # ===========================
    print(f"🔧 Defining Evaluation Step...")
    
    
    script_processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve('xgboost', region, version='1.5-1'),  # Use XGBoost container
        instance_type='ml.m5.xlarge',
        instance_count=1,
        base_job_name='flight-delay-evaluation',
        command=['python3']
    )    
    
    # Define property file for condition check
    evaluation_report = PropertyFile(
        name='evaluation',
        output_name='evaluation',
        path='evaluation.json'
    )
    
    evaluation_step = ProcessingStep(
        name='EvaluateModel',
        processor=script_processor,
        code=SCRIPT_PATH,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=TEST_DATA_PATH,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='evaluation',
                source='/opt/ml/processing/evaluation',
                destination=EVALUATION_OUTPUT_PATH
            )
        ],
        property_files=[evaluation_report]  
    )    
    # ===========================
    # STEP 3: CONDITION CHECK
    # ===========================
    print(f"🔧 Defining Condition Step...")
    
    f1_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file='evaluation',
            json_path='metrics.f1_score.value'
        ),
        right=0.05
    )
    
    print("   ✅ Condition: F1 >= 0.05")
    
    # ===========================
    # STEP 4: MODEL REGISTRATION
    # ===========================
    print(f"🔧 Defining Model Registration Step...")
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
            content_type='application/json'
        )
    )
    
    register_step = RegisterModel(
        name='RegisterFlightDelayModel',
        estimator=xgb_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.m5.xlarge'],
        transform_instances=['ml.m5.xlarge'],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status='PendingManualApproval',
        model_metrics=model_metrics
    )
    
    print("   ✅ Model registration step defined")
    
    # ===========================
    # STEP 5: CONDITIONAL STEP
    # ===========================
    print(f"🔧 Defining Conditional Step...")
    
    condition_step = ConditionStep(
        name='CheckF1Threshold',
        conditions=[f1_condition],
        if_steps=[register_step],
        else_steps=[]
    )
    
    print("   ✅ Conditional step defined")
    
    # ===========================
    # CREATE PIPELINE
    # ===========================
    print(f"\n🏗️  Creating Pipeline...")
    
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[max_depth, eta, num_round, subsample, colsample_bytree, scale_pos_weight, min_child_weight],
        steps=[training_step, evaluation_step, condition_step],
        sagemaker_session=session
    )
    
    return pipeline, role

# ===========================
# MAIN
# ===========================
if __name__ == '__main__':
    # Upload evaluation script
    upload_evaluation_script()
    
    # Create pipeline
    pipeline, role = create_pipeline()
    
    # Upsert pipeline to SageMaker
    print(f"\n📤 Uploading pipeline definition to SageMaker...")
    pipeline.upsert(role_arn=role)
    
    print(f"\n{'='*70}")
    print(f"🎉 PIPELINE CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nPipeline Name: {PIPELINE_NAME}")
    print(f"\n✅ View pipeline in SageMaker Studio")
    print(f"✅ Ready to run experiments with run_experiment.py")
    print(f"{'='*70}\n")
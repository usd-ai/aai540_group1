
import sys
import os
import json

# Add the directory containing settings_v2 to path
sys.path.append('/Users/arr/USD/AAI-540/aai540_group1/tmp/scripts_v2')

try:
    from pipeline_definition_v2 import create_pipeline
    
    print("Attempting to create pipeline definition...")
    pipeline, role = create_pipeline()
    
    definition = json.loads(pipeline.definition())
    
    print("\n✅ Pipeline definition created successfully!")
    print(f"Pipeline Name: {definition['Version']}") # Version 2020-12-01 usually
    
    print("\nParameters:")
    for param in definition['Parameters']:
        print(f" - {param['Name']} ({param['Type']}): {param.get('DefaultValue', 'No Default')}")
        
    print("\nSteps:")
    for step in definition['Steps']:
        print(f" - {step['Name']} ({step['Type']})")
        
except Exception as e:
    print(f"\n❌ Error creating pipeline: {e}")
    import traceback
    traceback.print_exc()

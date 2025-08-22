import json
import sys
from datetime import datetime

def generate_drift_report(model_version, drift_score):
    """Generate a comprehensive drift-aware model report"""
    
    # Load training metadata
    try:
        with open('training_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
    
    # Load drift report if available
    try:
        with open('drift_report.json', 'r') as f:
            drift_data = json.load(f)
    except FileNotFoundError:
        drift_data = {}
    
    # Create comprehensive report
    report = {
        'model_version': model_version,
        'report_generated': datetime.now().isoformat(),
        'drift_analysis': {
            'trigger_drift_score': float(drift_score),
            'drift_details': drift_data.get('data_drift', {}),
            'concept_drift_details': drift_data.get('concept_drift', {}),
            'semantic_drift_details': drift_data.get('semantic_drift', {}),
            'mitigation_applied': metadata.get('drift_info', {}).get('mitigation_applied', False)
        },
        'model_performance': {
            'accuracy': metadata.get('accuracy', 'N/A'),
            'training_samples': metadata.get('training_samples', 'N/A'),
            'test_samples': metadata.get('test_samples', 'N/A'),
            'vocabulary_size': metadata.get('vocabulary_size', 'N/A')
        },
        'model_parameters': metadata.get('model_parameters', {}),
        'recommendations': generate_recommendations(drift_score, drift_data),
        'status': 'completed',
        'files': [
            'spam_model.joblib',
            'vectorizer.joblib',
            'baseline_stats.json'
        ]
    }
    
    # Save report
    with open('drift_model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Drift-aware model report generated for version: {model_version}")
    print(f"Drift score that triggered retraining: {drift_score}")
    print(f"Model accuracy: {metadata.get('accuracy', 'N/A')}")

def generate_recommendations(drift_score, drift_data):
    """Generate recommendations based on drift analysis"""
    recommendations = []
    
    drift_score_float = float(drift_score)
    
    if drift_score_float > 0.3:
        recommendations.append("High drift detected - consider more frequent monitoring")
        recommendations.append("Evaluate data collection process for potential issues")
    elif drift_score_float > 0.15:
        recommendations.append("Moderate drift detected - monitor closely")
        recommendations.append("Consider feature engineering improvements")
    else:
        recommendations.append("Low drift detected - normal monitoring schedule sufficient")
    
    # Specific recommendations based on drift type
    if drift_data.get('data_drift', {}).get('detected', False):
        recommendations.append("Data distribution changed - review feature preprocessing")
    
    if drift_data.get('concept_drift', {}).get('detected', False):
        recommendations.append("Label distribution changed - review labeling criteria")
    
    if drift_data.get('semantic_drift', {}).get('detected', False):
        recommendations.append("Semantic patterns changed - consider vocabulary updates")
    
    return recommendations

if __name__ == "__main__":
    if len(sys.argv) > 2:
        model_version = sys.argv[1]
        drift_score = sys.argv[2]
    else:
        model_version = "unknown"
        drift_score = "0.0"
    
    generate_drift_report(model_version, drift_score)

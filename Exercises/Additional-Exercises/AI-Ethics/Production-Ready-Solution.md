# 🏥 **Solution: Operationalizing Ethics in Healthcare AI - Complete Code**

## 0️⃣ Setup \& Dataset Creation

```python
# Install required packages
!pip install --quiet scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Create the healthcare dataset
np.random.seed(456)

genders = ["Male", "Female", "Non-binary"]
ethnicities = ["White", "Black", "Hispanic", "Asian", "Other"]
insurance_types = ["Public", "Private", "Uninsured"]
conditions = ["Diabetes", "Hypertension", "Heart Disease", "Arthritis", "Mental Health"]

n = 1500
data = pd.DataFrame({
    "Patient_ID": range(1, n+1),
    "Age": np.random.randint(18, 85, n),
    "Gender": np.random.choice(genders, n, p=[0.48, 0.48, 0.04]),
    "Ethnicity": np.random.choice(ethnicities, n, p=[0.45, 0.2, 0.15, 0.15, 0.05]),
    "Insurance_Type": np.random.choice(insurance_types, n, p=[0.3, 0.6, 0.1]),
    "Primary_Condition": np.random.choice(conditions, n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
    "Severity_Score": np.random.randint(1, 11, n),
    "Previous_Treatments": np.random.randint(0, 5, n),
    "Comorbidities": np.random.randint(0, 4, n)
})

# Introduce bias patterns
data["Advanced_Treatment_Recommended"] = np.where(
    (data["Severity_Score"] >= 7) & (data["Insurance_Type"] == "Private"),
    np.random.choice([1, 0], n, p=[0.8, 0.2]),
    np.random.choice([1, 0], n, p=[0.4, 0.6])
)

# Additional ethnic bias
mask_bias = data["Ethnicity"].isin(["Black", "Hispanic"])
data.loc[mask_bias, "Advanced_Treatment_Recommended"] = np.where(
    mask_bias,
    np.random.choice([1, 0], mask_bias.sum(), p=[0.3, 0.7]),
    data.loc[mask_bias, "Advanced_Treatment_Recommended"]
)

print("Dataset created successfully!")
print(data.head())
```


***

## Part A — Development Ethics

### **Task 1 — Bias Detection in Development**

```python
# Prepare data for modeling
df_encoded = pd.get_dummies(data.drop(['Patient_ID'], axis=1), drop_first=True)
X = df_encoded.drop('Advanced_Treatment_Recommended', axis=1)
y = df_encoded['Advanced_Treatment_Recommended']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Bias detection by demographics
def analyze_bias_by_group(data, group_col):
    """Analyze treatment recommendation rates by demographic groups"""
    bias_analysis = data.groupby(group_col)['Advanced_Treatment_Recommended'].agg(['count', 'mean']).round(3)
    bias_analysis.columns = ['Total_Patients', 'Recommendation_Rate']
    return bias_analysis

print("=== BIAS ANALYSIS ===")
print("\nBy Gender:")
print(analyze_bias_by_group(data, 'Gender'))
print("\nBy Ethnicity:")
print(analyze_bias_by_group(data, 'Ethnicity'))
print("\nBy Insurance Type:")
print(analyze_bias_by_group(data, 'Insurance_Type'))

# Visualize bias
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['Gender', 'Ethnicity', 'Insurance_Type']):
    rates = data.groupby(col)['Advanced_Treatment_Recommended'].mean()
    axes[i].bar(rates.index, rates.values)
    axes[i].set_title(f'Treatment Recommendation Rate by {col}')
    axes[i].set_ylabel('Recommendation Rate')
    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.show()
```


### **Task 2 — Fairness Testing in Dev**

```python
def fairness_test(data, threshold=0.15):
    """Pre-deployment fairness test"""
    ethnicity_rates = data.groupby('Ethnicity')['Advanced_Treatment_Recommended'].mean()
    max_disparity = ethnicity_rates.max() - ethnicity_rates.min()
    
    print(f"=== FAIRNESS TEST ===")
    print(f"Maximum ethnic disparity: {max_disparity:.3f}")
    print(f"Threshold: {threshold}")
    
    if max_disparity > threshold:
        print("❌ FAIRNESS TEST FAILED - Model shows significant ethnic bias")
        return False
    else:
        print("✅ FAIRNESS TEST PASSED - Acceptable disparity levels")
        return True

fairness_result = fairness_test(data, threshold=0.15)
```


### **Task 3 — Feature Role \& Ethics Review**

```python
# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("=== FEATURE IMPORTANCE ANALYSIS ===")
print(feature_importance.head(10))

# Flag problematic features
problematic_features = []
for feature in feature_importance['Feature']:
    if any(demo in feature for demo in ['Gender', 'Ethnicity', 'Insurance']):
        problematic_features.append(feature)

print(f"\n⚠️  PROBLEMATIC FEATURES DETECTED: {problematic_features}")
print("\nRECOMMENDATIONS:")
print("1. Remove or downweight demographic features")
print("2. Focus on clinical features (Severity_Score, Comorbidities)")
print("3. Implement insurance-blind modeling")
```


### **Task 4 — Ethics-Aware Clinical User Stories**

```python
user_stories = [
    {
        "story": "As a physician, I want the AI to recommend treatments based purely on medical need, not insurance status",
        "acceptance_criteria": [
            "Insurance type should not be primary decision factor",
            "Clinical severity should dominate recommendations",
            "Model should pass insurance-bias test"
        ]
    },
    {
        "story": "As a hospital administrator, I want to ensure our AI provides equitable treatment recommendations across all ethnic groups",
        "acceptance_criteria": [
            "Ethnic disparity in recommendations < 15%",
            "Regular bias monitoring reports",
            "Alert system for equity violations"
        ]
    }
]

print("=== ETHICS-AWARE USER STORIES ===")
for i, story in enumerate(user_stories, 1):
    print(f"\nUser Story {i}:")
    print(f"Story: {story['story']}")
    print("Acceptance Criteria:")
    for criteria in story['acceptance_criteria']:
        print(f"  - {criteria}")
```


***

## Part B — Deployment Ethics

### **Task 5 — Clinical Decision Logging \& Accountability**

```python
import csv
import json

def log_treatment_recommendation(model, patient_data, model_version="v1.0", physician_override=None):
    """Log treatment recommendations for accountability"""
    
    # Make prediction
    prediction = model.predict(patient_data)[0]
    confidence = model.predict_proba(patient_data).max()
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id_hash": hashlib.sha256(str(patient_data.index).encode()).hexdigest()[:16],
        "age": int(patient_data.iloc['Age']) if 'Age' in patient_data.columns else "N/A",
        "severity_score": int(patient_data.iloc['Severity_Score']) if 'Severity_Score' in patient_data.columns else "N/A",
        "model_recommendation": int(prediction),
        "confidence_score": float(confidence),
        "model_version": model_version,
        "physician_override": physician_override
    }
    
    # Save to log file
    with open('treatment_recommendations_log.json', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    return log_entry

# Example logging
sample_patient = df_encoded.iloc[[0]]
log_result = log_treatment_recommendation(model, sample_patient)
print("=== SAMPLE LOG ENTRY ===")
print(json.dumps(log_result, indent=2))
```


### **Task 6 — Real-Time Health Equity Monitoring**

```python
class HealthEquityMonitor:
    def __init__(self, ethnicity_threshold=0.15, insurance_threshold=0.2):
        self.ethnicity_threshold = ethnicity_threshold
        self.insurance_threshold = insurance_threshold
        self.alerts = []
    
    def monitor_predictions(self, data_batch):
        """Monitor fairness in real-time predictions"""
        
        # Check ethnic disparities
        ethnic_rates = data_batch.groupby('Ethnicity')['Advanced_Treatment_Recommended'].mean()
        ethnic_disparity = ethnic_rates.max() - ethnic_rates.min()
        
        # Check insurance disparities
        insurance_rates = data_batch.groupby('Insurance_Type')['Advanced_Treatment_Recommended'].mean()
        insurance_disparity = insurance_rates.max() - insurance_rates.min()
        
        # Generate alerts
        if ethnic_disparity > self.ethnicity_threshold:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "ethnic_bias_alert",
                "disparity": ethnic_disparity,
                "threshold": self.ethnicity_threshold
            }
            self.alerts.append(alert)
            print(f"🚨 ETHNIC BIAS ALERT: Disparity {ethnic_disparity:.3f} exceeds threshold {self.ethnicity_threshold}")
        
        if insurance_disparity > self.insurance_threshold:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "insurance_bias_alert",
                "disparity": insurance_disparity,
                "threshold": self.insurance_threshold
            }
            self.alerts.append(alert)
            print(f"🚨 INSURANCE BIAS ALERT: Disparity {insurance_disparity:.3f} exceeds threshold {self.insurance_threshold}")
        
        return len(self.alerts) == 0  # Return True if no alerts

# Test monitoring
monitor = HealthEquityMonitor()
monitoring_result = monitor.monitor_predictions(data.sample(100))
print(f"\nMonitoring Status: {'✅ All Clear' if monitoring_result else '⚠️ Alerts Generated'}")
```


### **Task 7 — Patient Privacy Controls**

```python
def anonymize_patient_data(data):
    """Apply medical-grade anonymization"""
    anonymized_data = data.copy()
    
    # Hash patient IDs
    anonymized_data['Patient_ID_Hash'] = data['Patient_ID'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
    )
    
    # Remove original ID
    anonymized_data = anonymized_data.drop('Patient_ID', axis=1)
    
    # Age binning for additional privacy
    anonymized_data['Age_Group'] = pd.cut(
        anonymized_data['Age'], 
        bins=[0, 30, 50, 70, 100], 
        labels=['18-30', '31-50', '51-70', '70+']
    )
    
    return anonymized_data

anonymized_sample = anonymize_patient_data(data.head())
print("=== ANONYMIZED DATA SAMPLE ===")
print(anonymized_sample[['Patient_ID_Hash', 'Age_Group', 'Gender', 'Ethnicity']].head())
```


### **Task 8 — Health Equity Feedback Loop**

```python
def collect_provider_feedback():
    """Simulate collecting feedback from healthcare providers"""
    
    # Simulate feedback data
    feedback_scenarios = [
        {
            "provider_id": "DR001",
            "patient_demographic": "Black, Female, Public Insurance",
            "ai_recommendation": "No Advanced Treatment",
            "clinical_outcome": "Poor - Required Emergency Care",
            "provider_assessment": "AI recommendation was inappropriate for severity level",
            "suggested_improvement": "Consider clinical severity over insurance status"
        },
        {
            "provider_id": "DR002", 
            "patient_demographic": "Hispanic, Male, Uninsured",
            "ai_recommendation": "No Advanced Treatment",
            "clinical_outcome": "Poor - Condition Worsened",
            "provider_assessment": "Economic factors shouldn't influence medical recommendations",
            "suggested_improvement": "Implement insurance-blind clinical protocols"
        }
    ]
    
    return feedback_scenarios

def process_feedback_for_retraining(feedback_data):
    """Process feedback to improve model"""
    
    print("=== PROVIDER FEEDBACK ANALYSIS ===")
    for i, feedback in enumerate(feedback_data, 1):
        print(f"\nFeedback {i}:")
        print(f"Issue: {feedback['provider_assessment']}")
        print(f"Improvement: {feedback['suggested_improvement']}")
    
    recommendations = [
        "Increase weight of clinical features (severity, comorbidities)",
        "Reduce or eliminate insurance-based features", 
        "Retrain model with balanced demographic representation",
        "Implement clinical override protocols"
    ]
    
    print(f"\n=== RETRAINING RECOMMENDATIONS ===")
    for rec in recommendations:
        print(f"• {rec}")

feedback_data = collect_provider_feedback()
process_feedback_for_retraining(feedback_data)
```


### **Task 9 — Medical Ethics Incident Simulation**

```python
class MedicalEthicsIncidentResponse:
    def __init__(self):
        self.incident_log = []
    
    def handle_incident(self, incident_type, description, affected_demographics):
        """Handle medical ethics incidents"""
        
        incident = {
            "incident_id": f"MED-ETH-{len(self.incident_log) + 1:04d}",
            "timestamp": datetime.now().isoformat(),
            "type": incident_type,
            "description": description,
            "affected_demographics": affected_demographics,
            "status": "Under Investigation"
        }
        
        print(f"=== MEDICAL ETHICS INCIDENT RESPONSE ===")
        print(f"Incident ID: {incident['incident_id']}")
        print(f"Type: {incident_type}")
        print(f"Description: {description}")
        
        # Step 1: Detection & Clinical Review
        print(f"\n🔍 STEP 1: Detection & Clinical Review")
        print("- Incident logged in ethics system")
        print("- Chief Medical Officer notified")
        print("- Clinical data review initiated")
        
        # Step 2: Investigation with Ethics Committee
        print(f"\n🏥 STEP 2: Investigation with Medical Ethics Committee")
        print("- Ethics committee convened")
        print("- Statistical analysis of affected demographics")
        print("- Clinical expert review of cases")
        
        # Step 3: Patient/Provider Communication
        print(f"\n📞 STEP 3: Patient/Provider Communication")
        print("- Affected patients notified (where appropriate)")
        print("- Healthcare providers briefed on findings")
        print("- Transparent communication plan activated")
        
        # Step 4: Clinical Remediation
        print(f"\n⚕️ STEP 4: Clinical Remediation & Policy Updates")
        print("- AI model temporarily suspended for high-risk cases")
        print("- Clinical oversight protocols enhanced")
        print("- Model retraining with corrected data initiated")
        print("- Policy updates to prevent recurrence")
        
        self.incident_log.append(incident)
        return incident

# Simulate incident
incident_handler = MedicalEthicsIncidentResponse()
incident = incident_handler.handle_incident(
    "Healthcare Disparity",
    "AI model showing 40% lower advanced treatment recommendations for Black and Hispanic patients with equivalent clinical severity",
    ["Black patients", "Hispanic patients"]
)
```


### **Bonus — Clinical Transparency Report**

```python
def generate_clinical_transparency_report():
    """Generate transparency report for stakeholders"""
    
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "model_version": "v1.0",
        "reporting_period": "Q1 2025",
        "total_patients_analyzed": len(data),
        
        "fairness_metrics": {
            "ethnic_disparity": data.groupby('Ethnicity')['Advanced_Treatment_Recommended'].mean().max() - 
                              data.groupby('Ethnicity')['Advanced_Treatment_Recommended'].mean().min(),
            "insurance_disparity": data.groupby('Insurance_Type')['Advanced_Treatment_Recommended'].mean().max() - 
                                 data.groupby('Insurance_Type')['Advanced_Treatment_Recommended'].mean().min()
        },
        
        "model_performance": {
            "accuracy": model.score(X_test, y_test),
            "top_clinical_features": feature_importance.head(3)['Feature'].tolist()
        },
        
        "recommendations": [
            "Reduce insurance-based feature importance",
            "Implement regular bias monitoring",
            "Establish clinical override protocols",
            "Enhance provider training on AI ethics"
        ]
    }
    
    print("=== CLINICAL TRANSPARENCY REPORT ===")
    print(json.dumps(report, indent=2, default=str))
    
    return report

transparency_report = generate_clinical_transparency_report()
```


***

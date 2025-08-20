# Solution

Here is a complete Python script solution for the customer churn ML metadata tracking exercise that runs locally with a synthetic dataset. It covers dataset generation, training, metadata registration, logging, visualization, and lineage exploration.

***

# Directory structure to create beforehand

```
customer_churn_metadata/
├── data/
├── models/
```

Create the folders `data` and `models` inside your `customer_churn_metadata` directory.

***

# Script: run_churn_metadata.py

```python
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# --- Setup directories ---
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- SQLAlchemy base and ORM models ---
Base = declarative_base()

class Artifact(Base):
    __tablename__ = 'artifacts'
    id = Column(Integer, primary_key=True)
    type = Column(String)           # 'DataSet', 'SavedModel', etc.
    uri = Column(String)
    version = Column(Integer)
    name = Column(String, nullable=True)  # for models
    split = Column(String, nullable=True) # for datasets

class Execution(Base):
    __tablename__ = 'executions'
    id = Column(Integer, primary_key=True)
    type = Column(String)  # e.g. 'Trainer'
    state = Column(String) # e.g. 'RUNNING', 'COMPLETED'

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    artifact_id = Column(Integer, ForeignKey('artifacts.id'))
    execution_id = Column(Integer, ForeignKey('executions.id'))
    type = Column(String)  # 'INPUT' or 'OUTPUT'

    artifact = relationship('Artifact')
    execution = relationship('Execution')

# --- Initialize SQLite DB and session ---
def init_metadata_store(db_url='sqlite:///metadata.db'):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    print("Initialized local metadata store (SQLite).")
    return Session()

# --- Generate synthetic churn dataset and save ---
def generate_and_save_churn_dataset():
    print("Generating synthetic customer churn dataset...")
    np.random.seed(42)

    n_samples = 100
    customer_id = [f"CUST{i:04d}" for i in range(n_samples)]
    monthly_charges = np.random.uniform(20, 120, n_samples).round(2)
    tenure = np.random.randint(1, 60, n_samples)
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                      n_samples, p=[0.6, 0.25, 0.15])
    churn_prob = np.where(contract_types == 'Month-to-month', 0.4, 0.1) + (tenure < 12) * 0.2
    churn = np.random.binomial(1, churn_prob.clip(0,1))

    df = pd.DataFrame({
        'customer_id': customer_id,
        'monthly_charges': monthly_charges,
        'tenure': tenure,
        'contract_type': contract_types,
        'churn': churn
    })

    data_path = os.path.join(DATA_DIR, 'churn_dataset.csv')
    df.to_csv(data_path, index=False)
    print(f"Synthetic churn dataset saved to {data_path}")
    return data_path

# --- Metadata tracking functions ---
def register_dataset(session, data_path, version=1, split='full'):
    artifact = Artifact(
        type='DataSet',
        uri=f'file://{os.path.abspath(data_path)}',
        version=version,
        split=split
    )
    session.add(artifact)
    session.commit()
    print(f"Registered dataset artifact ID: {artifact.id}")
    return artifact

def register_model(session, model_path, name='ChurnPredictionModel', version=1):
    artifact = Artifact(
        type='SavedModel',
        uri=f'file://{os.path.abspath(model_path)}',
        name=name,
        version=version
    )
    session.add(artifact)
    session.commit()
    print(f"Registered model artifact ID: {artifact.id}")
    return artifact

def register_execution(session, exec_type='Trainer', state='RUNNING'):
    execution = Execution(type=exec_type, state=state)
    session.add(execution)
    session.commit()
    print(f"Started training execution ID: {execution.id} with state: {state}")
    return execution

def link_artifact_execution(session, artifact_id, execution_id, event_type):
    event = Event(artifact_id=artifact_id, execution_id=execution_id, type=event_type)
    session.add(event)
    session.commit()

def mark_execution_completed(session, execution):
    execution.state = 'COMPLETED'
    session.commit()
    print(f"Training execution {execution.id} marked as COMPLETED.")

# --- Train churn prediction model and log metadata ---
def train_and_log(session, dataset_artifact, dataset_path):
    print("Training churn prediction model...")

    df = pd.read_csv(dataset_path)
    X = df[['monthly_charges', 'tenure', 'contract_type']]
    y = df['churn']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(sparse=False), ['contract_type']),
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    pipeline.fit(X, y)

    model_path = os.path.join(MODEL_DIR, 'churn_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")

    execution = register_execution(session, exec_type='Trainer', state='RUNNING')

    link_artifact_execution(session, dataset_artifact.id, execution.id, 'INPUT')

    model_artifact = register_model(session, model_path)
    link_artifact_execution(session, model_artifact.id, execution.id, 'OUTPUT')

    mark_execution_completed(session, execution)

    return execution.id

# --- Visualize execution states ---
def visualize_metadata(session):
    executions = session.query(Execution).all()
    states = [e.state for e in executions]
    counts = Counter(states)

    print("Execution states frequency:")
    for state, count in counts.items():
        print(f"  {state}: {count}")

    plt.bar(counts.keys(), counts.values())
    plt.title('Execution States Count')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.show()

# --- Print lineage info for a given execution ---
def print_lineage(session, execution_id):
    print(f"\nLineage information for execution ID: {execution_id}")
    events = session.query(Event).filter(Event.execution_id == execution_id).all()

    input_artifacts = [e.artifact for e in events if e.type == 'INPUT']
    output_artifacts = [e.artifact for e in events if e.type == 'OUTPUT']

    print("Input Artifacts:")
    for a in input_artifacts:
        print(f"  ID: {a.id}, URI: {a.uri}")

    print("Output Artifacts:")
    for a in output_artifacts:
        print(f"  ID: {a.id}, URI: {a.uri}")

# --- Main workflow ---
def main():
    session = init_metadata_store()
    data_path = generate_and_save_churn_dataset()
    data_artifact = register_dataset(session, data_path)
    execution_id = train_and_log(session, data_artifact, data_path)
    visualize_metadata(session)
    print_lineage(session, execution_id)

if __name__ == '__main__':
    main()
```


***

## Instructions to Run

1. Inside your working directory, create the folders:
```
customer_churn_metadata/
├── data/
├── models/
```

2. Save the above code as `run_churn_metadata.py` inside `customer_churn_metadata`.
3. From the command line, navigate into the `customer_churn_metadata` folder.
4. Install dependencies if needed:
```bash
pip install ml-metadata pandas scikit-learn matplotlib
```

5. Run the script:
```bash
python run_churn_metadata.py
```


***

## What This Does

- Generates a synthetic customer churn dataset and saves as CSV.
- Registers dataset and model types in the ML metadata store (local SQLite `metadata.db`).
- Trains a logistic regression churn model using one-hot encoding for contract type.
- Saves the trained model.
- Logs the dataset artifact, training execution (from RUNNING to COMPLETED), and output model artifact in the metadata store.
- Displays a bar plot showing counts of executions by state.
- Prints detailed lineage info showing which datasets and models correspond to the logged execution.

***

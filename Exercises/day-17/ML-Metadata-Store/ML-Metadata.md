# ML Metadata Store Exercise: Tracking ML Workflow from Data to Model with Visualization

This exercise will guide you through building a local machine learning metadata tracking system using the TensorFlow ML Metadata (MLMD) library. You will simulate data collection, train a model, and record all relevant metadata (datasets, executions, models) in a local SQLite database. Finally, you will visualize execution metadata and explore lineage information.

***

## Exercise Overview

- **Objective:** Set up a local ML metadata store and track the complete ML workflow metadata.
- **Stack:** Python, ML Metadata (MLMD), scikit-learn, pandas, matplotlib, SQLite.
- **Outcome:** Save dataset and model, record metadata, visualize metadata states, and inspect lineage.
- **Directory Structure:** Organized folders for data, models, and metadata DB.

***

## Directory Structure Setup

```
ml_metadata_exercise/
├── data/                      # Stores dataset CSV files
├── models/                    # Stores saved ML models
├── metadata.db                # SQLite metadata store (auto-created)
└── run_metadata_exercise.py   # Complete Python script
```

Create the folders `data/` and `models/` inside your `ml_metadata_exercise` directory before running the script.

***

## Step-by-Step Python Script: run_metadata_exercise.py

```python
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
from collections import Counter

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
    type = Column(String)           # e.g. 'DataSet', 'SavedModel'
    uri = Column(String)
    version = Column(Integer)
    name = Column(String, nullable=True)
    split = Column(String, nullable=True)  # For datasets: e.g. 'train'

class Execution(Base):
    __tablename__ = 'executions'
    id = Column(Integer, primary_key=True)
    type = Column(String)  # e.g. 'Trainer'
    state = Column(String)  # e.g. 'RUNNING', 'COMPLETED'

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    artifact_id = Column(Integer, ForeignKey('artifacts.id'))
    execution_id = Column(Integer, ForeignKey('executions.id'))
    type = Column(String)  # 'INPUT' or 'OUTPUT'

    artifact = relationship('Artifact')
    execution = relationship('Execution')

# --- Initialize SQLite database and session ---
def init_metadata_store(db_url='sqlite:///metadata.db'):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    print("Initialized local metadata store (SQLite).")
    return Session()

# --- Create and save a simple dataset ---
def create_and_save_dataset():
    print("Creating sample dataset...")
    data = pd.DataFrame({
        'feature1': range(10),
        'feature2': [x * 2 for x in range(10)],
        'label': [0, 1] * 5
    })
    data_path = os.path.join(DATA_DIR, 'train_dataset.csv')
    data.to_csv(data_path, index=False)
    print(f"Dataset saved at {data_path}")
    return data_path

# --- Register dataset artifact ---
def register_dataset(session, data_path, version=1, split='train'):
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

# --- Register model artifact ---
def register_model(session, model_path, name='LogisticRegressionModel', version=1):
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

# --- Register execution ---
def register_execution(session, exec_type='Trainer', state='RUNNING'):
    execution = Execution(type=exec_type, state=state)
    session.add(execution)
    session.commit()
    print(f"Started execution ID: {execution.id} with state: {state}")
    return execution

# --- Link artifact and execution through events ---
def link_artifact_execution(session, artifact_id, execution_id, event_type):
    event = Event(artifact_id=artifact_id, execution_id=execution_id, type=event_type)
    session.add(event)
    session.commit()

# --- Mark execution as COMPLETED ---
def mark_execution_completed(session, execution):
    execution.state = 'COMPLETED'
    session.commit()
    print(f"Execution {execution.id} marked as COMPLETED.")

# --- Train model and log metadata ---
def train_and_log(session, dataset_artifact, dataset_path):
    print("Training model...")
    df = pd.read_csv(dataset_path)
    X = df[['feature1', 'feature2']]
    y = df['label']

    model = LogisticRegression()
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, 'trained_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    execution = register_execution(session, exec_type='Trainer', state='RUNNING')

    # Link dataset as input artifact
    link_artifact_execution(session, dataset_artifact.id, execution.id, 'INPUT')

    # Register and link model artifact as output
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
    plt.title('Executions by State')
    plt.xlabel('Execution State')
    plt.ylabel('Count')
    plt.show()

# --- Print lineage for a given execution ---
def print_lineage(session, execution_id):
    print(f"\nLineage info for execution ID: {execution_id}")
    events = session.query(Event).filter(Event.execution_id == execution_id).all()

    input_artifacts = [e.artifact for e in events if e.type == 'INPUT']
    output_artifacts = [e.artifact for e in events if e.type == 'OUTPUT']

    print("Input Artifacts:")
    for a in input_artifacts:
        print(f"  ID: {a.id}, URI: {a.uri}")

    print("Output Artifacts:")
    for a in output_artifacts:
        print(f"  ID: {a.id}, URI: {a.uri}")

# --- Main ---
def main():
    session = init_metadata_store()
    dataset_path = create_and_save_dataset()
    dataset_artifact = register_dataset(session, dataset_path)
    execution_id = train_and_log(session, dataset_artifact, dataset_path)
    visualize_metadata(session)
    print_lineage(session, execution_id)

if __name__ == "__main__":
    main()
```


***

## How to Run This Exercise

1. Create a folder named `ml_metadata_exercise`.
2. Inside it, create two folders: `data` and `models`.
3. Save the above script as `run_metadata_exercise.py` in `ml_metadata_exercise`.
4. Open a terminal or command prompt, navigate to the `ml_metadata_exercise` folder.
5. Install dependencies if not already installed:
```
pip install ml-metadata pandas scikit-learn matplotlib
```

6. Run the script:
```
python run_metadata_exercise.py
```


***

## Viewing and Exploring the Metadata Store

- The metadata is saved in the SQLite file `metadata.db` inside `ml_metadata_exercise`.
- You can browse this database using any SQLite viewer, for example:
    - **DB Browser for SQLite** (GUI tool)
    - Command line:

```bash
sqlite3 metadata.db
```

Then run queries such as:

```sql
.tables
SELECT * FROM artifact;  
SELECT * FROM execution;
SELECT * FROM event;
```

- This lets you inspect stored artifacts (datasets/models), executions (training runs), and their relationships.

***

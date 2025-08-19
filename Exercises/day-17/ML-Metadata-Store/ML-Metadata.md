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

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

# --- Setup directories ---
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Initialize local ML Metadata store with SQLite ---
def init_metadata_store():
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = 'metadata.db'
    connection_config.sqlite.connection_mode = metadata_store_pb2.SqliteConfig.READWRITE_OPENCREATE
    store = metadata_store.MetadataStore(connection_config)
    print("Initialized local ML metadata store.")
    return store

# --- Create and save a simple training dataset ---
def create_and_save_dataset():
    print("Creating sample dataset...")
    data = pd.DataFrame({
        'feature1': range(10),
        'feature2': [x * 2 for x in range(10)],
        'label': [0, 1]*5
    })
    data_path = os.path.join(DATA_DIR, 'train_dataset.csv')
    data.to_csv(data_path, index=False)
    print(f"Dataset saved at {data_path}")
    return data_path

# --- Register metadata types (Artifact and Execution) ---
def register_types(store):
    data_type = metadata_store_pb2.ArtifactType(name='DataSet')
    data_type.properties['version'] = metadata_store_pb2.INT
    data_type.properties['split'] = metadata_store_pb2.STRING
    data_type_id = store.put_artifact_type(data_type)

    model_type = metadata_store_pb2.ArtifactType(name='SavedModel')
    model_type.properties['version'] = metadata_store_pb2.INT
    model_type.properties['name'] = metadata_store_pb2.STRING
    model_type_id = store.put_artifact_type(model_type)

    trainer_type = metadata_store_pb2.ExecutionType(name='Trainer')
    trainer_type.properties['state'] = metadata_store_pb2.STRING
    trainer_type_id = store.put_execution_type(trainer_type)

    print("Registered metadata types.")
    return data_type_id, model_type_id, trainer_type_id

# --- Register dataset artifact in metadata store ---
def register_dataset(store, data_type_id, data_path):
    data_artifact = metadata_store_pb2.Artifact()
    data_artifact.type_id = data_type_id
    data_artifact.uri = f'file://{os.path.abspath(data_path)}'
    data_artifact.properties['version'].int_value = 1
    data_artifact.properties['split'].string_value = 'train'
    [registered_data] = store.put_artifacts([data_artifact])
    print(f"Registered dataset artifact ID: {registered_data.id}")
    return registered_data

# --- Train ML model and log training execution and artifacts ---
def train_and_log(store, trainer_type_id, model_type_id, data_artifact, data_path):
    print("Training model...")
    df = pd.read_csv(data_path)
    X = df[['feature1', 'feature2']]
    y = df['label']

    model = LogisticRegression()
    model.fit(X, y)
    
    model_path = os.path.join(MODEL_DIR, 'trained_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # Log execution: start RUNNING
    execution = metadata_store_pb2.Execution()
    execution.type_id = trainer_type_id
    execution.properties['state'].string_value = 'RUNNING'
    [execution] = store.put_executions([execution])
    print(f"Started training execution ID: {execution.id}")

    # Link input data artifact
    input_event = metadata_store_pb2.Event()
    input_event.artifact_id = data_artifact.id
    input_event.execution_id = execution.id
    input_event.type = metadata_store_pb2.Event.INPUT
    store.put_events([input_event])

    # Register trained model artifact
    model_artifact = metadata_store_pb2.Artifact()
    model_artifact.type_id = model_type_id
    model_artifact.uri = f'file://{os.path.abspath(model_path)}'
    model_artifact.properties['version'].int_value = 1
    model_artifact.properties['name'].string_value = 'LogisticRegressionModel'
    [model_artifact] = store.put_artifacts([model_artifact])
    print(f"Registered model artifact ID: {model_artifact.id}")

    # Link output model artifact to execution
    output_event = metadata_store_pb2.Event()
    output_event.artifact_id = model_artifact.id
    output_event.execution_id = execution.id
    output_event.type = metadata_store_pb2.Event.OUTPUT
    store.put_events([output_event])

    # Mark execution as COMPLETED
    execution.properties['state'].string_value = 'COMPLETED'
    store.put_executions([execution])
    print(f"Execution {execution.id} marked as COMPLETED.")
    return execution.id

# --- Visualize execution metadata ---
def visualize_metadata(store):
    executions = store.get_executions()
    states = [e.properties['state'].string_value for e in executions]
    counts = Counter(states)

    print("Execution states frequency:")
    for state, count in counts.items():
        print(f"  {state}: {count}")

    plt.bar(counts.keys(), counts.values())
    plt.title('Executions by State')
    plt.xlabel('Execution State')
    plt.ylabel('Count')
    plt.show()

# --- Print artifacts and lineage for a given execution ---
def print_lineage(store, execution_id):
    print(f"\nLineage info for execution ID: {execution_id}")
    events = store.get_events_by_execution(execution_id)
    input_artifact_ids = [e.artifact_id for e in events if e.type == metadata_store_pb2.Event.INPUT]
    output_artifact_ids = [e.artifact_id for e in events if e.type == metadata_store_pb2.Event.OUTPUT]

    print("Input Artifacts:")
    for aid in input_artifact_ids:
        artifact = store.get_artifacts_by_id([aid])[0]
        print(f"  ID: {artifact.id}, URI: {artifact.uri}")

    print("Output Artifacts:")
    for aid in output_artifact_ids:
        artifact = store.get_artifacts_by_id([aid])
        print(f"  ID: {artifact.id}, URI: {artifact.uri}")

# --- Main method ---
def main():
    store = init_metadata_store()
    data_path = create_and_save_dataset()
    data_type_id, model_type_id, trainer_type_id = register_types(store)
    data_artifact = register_dataset(store, data_type_id, data_path)
    execution_id = train_and_log(store, trainer_type_id, model_type_id, data_artifact, data_path)
    visualize_metadata(store)
    print_lineage(store, execution_id)

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

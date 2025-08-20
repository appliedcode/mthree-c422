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
    print("Initialized local ML Metadata store.")
    return store

# --- Generate synthetic churn dataset ---
def generate_and_save_churn_dataset():
    print("Generating synthetic customer churn dataset...")

    # Parameters for synthetic dataset
    n_samples = 100
    import numpy as np
    np.random.seed(42)
    
    customer_id = [f"CUST{i:04d}" for i in range(n_samples)]
    monthly_charges = np.random.uniform(20, 120, n_samples).round(2)
    tenure = np.random.randint(1, 60, n_samples)
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.25, 0.15])
    # Generate churn: higher churn with month-to-month contracts and low tenure
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

# --- Register metadata types ---
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

# --- Register dataset artifact ---
def register_dataset(store, data_type_id, data_path):
    data_artifact = metadata_store_pb2.Artifact()
    data_artifact.type_id = data_type_id
    data_artifact.uri = f'file://{os.path.abspath(data_path)}'
    data_artifact.properties['version'].int_value = 1
    data_artifact.properties['split'].string_value = 'full'
    [registered_data] = store.put_artifacts([data_artifact])
    print(f"Registered dataset artifact ID: {registered_data.id}")
    return registered_data

# --- Train model and log metadata ---
def train_and_log(store, trainer_type_id, model_type_id, data_artifact, data_path):
    print("Training churn prediction model...")

    df = pd.read_csv(data_path)
    X = df[['monthly_charges', 'tenure', 'contract_type']]
    y = df['churn']

    # Preprocessing pipeline with one-hot encoding for contract_type
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

    # Start execution with RUNNING state
    execution = metadata_store_pb2.Execution()
    execution.type_id = trainer_type_id
    execution.properties['state'].string_value = 'RUNNING'
    [execution] = store.put_executions([execution])
    print(f"Started training execution ID: {execution.id}")

    # Link input dataset artifact
    input_event = metadata_store_pb2.Event()
    input_event.artifact_id = data_artifact.id
    input_event.execution_id = execution.id
    input_event.type = metadata_store_pb2.Event.INPUT
    store.put_events([input_event])

    # Register model artifact
    model_artifact = metadata_store_pb2.Artifact()
    model_artifact.type_id = model_type_id
    model_artifact.uri = f'file://{os.path.abspath(model_path)}'
    model_artifact.properties['version'].int_value = 1
    model_artifact.properties['name'].string_value = 'ChurnPredictionModel'
    [model_artifact] = store.put_artifacts([model_artifact])
    print(f"Registered model artifact ID: {model_artifact.id}")

    # Link model artifact as output of training execution
    output_event = metadata_store_pb2.Event()
    output_event.artifact_id = model_artifact.id
    output_event.execution_id = execution.id
    output_event.type = metadata_store_pb2.Event.OUTPUT
    store.put_events([output_event])

    # Complete execution
    execution.properties['state'].string_value = 'COMPLETED'
    store.put_executions([execution])
    print(f"Training execution {execution.id} marked as COMPLETED.")

    return execution.id

# --- Visualize metadata (execution states counts) ---
def visualize_metadata(store):
    executions = store.get_executions()
    states = [e.properties['state'].string_value for e in executions]
    counts = Counter(states)
    print("Execution states frequency:")
    for state, count in counts.items():
        print(f"  {state}: {count}")

    plt.bar(counts.keys(), counts.values())
    plt.title('Execution States Count')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.show()

# --- Print lineage for execution (input/output artifacts) ---
def print_lineage(store, execution_id):
    print(f"\nLineage information for execution ID: {execution_id}")
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

# --- Main workflow ---
def main():
    store = init_metadata_store()
    data_path = generate_and_save_churn_dataset()
    data_type_id, model_type_id, trainer_type_id = register_types(store)
    data_artifact = register_dataset(store, data_type_id, data_path)
    execution_id = train_and_log(store, trainer_type_id, model_type_id, data_artifact, data_path)
    visualize_metadata(store)
    print_lineage(store, execution_id)

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

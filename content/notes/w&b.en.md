Title: Weights & Biases platform
Date: 2025-08-04 12:15
Category: MLOps
Lang: en
Slug: mlops_1
Author: Facundo Roffet

<!-- Hide default title -->
<style> h1.entry-title, h1.post-title, h1.title, h1:first-of-type {display: none;} </style>
<!-- Add custom title -->
<h2 style="text-align: center; font-size: 3em; color: rgba(12, 205, 76, 0.927);">Weights & Biases platform</h2>

<!---------------------------------------------------------------------------->

> These are my personal notes from the courses [Weights & Biases 101](https://wandb.ai/site/courses/101/), [Weights & Biases 201: Model Registry](https://www.wandb.courses/courses/201-model-registry), [Weights & Biases 101: Weave](https://wandb.ai/site/courses/weave/) and [Effective MLOps: Model development](https://wandb.ai/site/courses/effective-mlops/).

<!---------------------------------------------------------------------------->

## Training AI models: W\&B Models

#### Runs
* A Run is a unit of Python code execution that captures its entire context (library versions, hardware, system metrics, Git state, etc).
* It can include various types of logs: scalar metrics, media (images, histograms, 3D visualizations, videos, audio), plots, tables, and more.

#### Projects
* A Project contains multiple Runs and all their associated data.
* In the GUI, a Project has different tabs:
  * Overview: general information.
  * Workspace: interactive dashboard with customizable panels to visualize results and compare Runs.
  * Runs: table listing all the Runs in the Project.
  * Automations: configured automated processes.
  * Sweeps: a set of Runs that automatically explore different hyperparameters.
  * Reports: flexible, auto-updating, interactive documents used to share visualizations from a Project.

#### Artifacts
* An Artifact is a way to track and version any data used as input or output of a Run: datasets, evaluation results, or model checkpoints. It encapsulates both files and folders, and is composed of:
  * A name
  * A type (most useful types are 'dataset' and 'model')
  * Metadata (as a dictionary)
  * A description
  * Files, folders, or references to external object storage (e.g., S3)
* Each time an Artifact with the same name is logged, it's compared to previous versions. If any difference is found, a new version is created with an alias (v1, v2, v3, etc).
* In the W\&B GUI you can view version info, metadata, contained files, lineage (a tree of processes that created or used the Artifact), and usage snippets.

#### Model Registry
* The Model Registry is a centralized repository that hosts and organizes different Collections (also called Model Tasks).
* In other words, the Model Registry is where all models ready for public use live—either for testing or production.
* A Collection refers to a particular task (e.g., building segmentation, cell classification). It contains 'model'-type Artifacts that have been promoted as candidates for use by other teams or processes. Personal opinion: each Project should have a single associated Collection.
* A Collection can contain multiple versions of the same model, each with its own alias—for example: staging and production.
* Automations can be set up in the Model Registry—these are automatic actions triggered when a new model or alias is added (e.g., generating a Report).

<!---------------------------------------------------------------------------->

## Developing applications with LLMs: W\&B Weave

#### Calls
* Calls are the fundamental building block in Weave, representing a single function execution, including its inputs (arguments), outputs (returns), and metadata (timing, exceptions, etc).
* A Call can be related to another via parent/child relationships, forming a tree structure.
* Once created, a Call cannot be modified. You can only add feedback or delete it (both via code or GUI).

#### Traces
* A Trace (or Span) is a collection of Calls that share the same execution context.

#### Objects (Models and Datasets) and Ops
* An Object is a data type that Weave can understand, serialize, and version. Models and Datasets are subclasses of Objects; Ops are functions/methods that operate on Objects.
* A Model allows you to version an app over time and understand how changes affect its responses. It consists of data (config, checkpoint, etc.) and code that defines how the model operates.
* To create a Model, define a class that inherits from `weave.Model`, specify type annotations for all fields, and implement an `invoke` or `predict` method decorated with `@weave.op`.
* When model-defining parameters change, the changes are logged and the model version is automatically updated.
* A Dataset is a collection of examples used to evaluate a Model.

#### Evaluations
* An Evaluation assesses the performance of a Model on a Dataset using a specified list of metrics (functions or Scorers).
* Latency and token count are metrics logged by default.
* In the GUI, you can perform in-depth comparisons between different Evaluations.

<!---------------------------------------------------------------------------->

## Basic usage in code

#### Installation
```
pip install wandb weave
```

#### Setup
```python
import wandb, weave
wandb.login()
@weave.op()  # decorator for every function/method to track (automatically included in common LLM libraries)
```

#### Initialization
```python
run = wandb.init(project=project_name, entity=entity_name, job_type=job_type, config=dict_with_configs)
client = weave.init(project_name)  # only needed when using weave as standalone
```

#### Logging
Scalars and files:
```python
run.log({'train/epoch': epoch, 'train/train_loss': train_loss})
run.log({'val/val_loss': val_loss, 'val/accuracy': acc})
run.log_model(path_to_model_file)  # same as creating an Artifact of type 'model'
```
Artifacts:
```python
# Log
artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
artifact.add_file(path_to_file)
artifact.add_dir(path_to_folder)
run.log_artifact(artifact)
# Retrieve
artifact = run.use_artifact(f"{artifact_name}:{artifact_version}")
artifact_dir = artifact.download()
```
Objects:
```python
# Log
weave.publish(variable_to_log, log_name)
# Retrieve
object = weave.ref(f"{object_name}:{object_version}").get()
```

#### Datasets

```python
# Manual
dataset = Dataset(
    name=dataset_name,
    rows=[
        {'id': id1, 'sentence': query1, 'correction': gt1},
        {'id': id2, 'sentence': query2, 'correction': gt2}
    ]
)
# From Calls
dataset = Dataset.from_calls(call_list)
# From DataFrame
dataset = Dataset.from_pandas(df)
```

#### Evaluations
```python
evaluation = weave.Evaluation(dataset=dataset, scorers=scorers_list)
evaluation.evaluate(model)
```

#### Finalization
```python
# Only needed in notebooks
run.finish()
```

Aquí tenés la traducción al inglés de la nueva sección, adaptada al estilo del resto del documento:

<!---------------------------------------------------------------------------->

## Best practices pipeline

#### 1. Explore the data
* Initialize a Run with `job_type='data_upload'`.
* Create an Artifact with `type='raw_data'`.
* Add all files and folders to the Artifact.
* Create a Table with the data and add it to the Artifact.
* Create a Report with the Table and annotate key findings.

#### 2. Create a dataset
* Initialize a Run with `job_type='data_processing'`.
* Download the raw data Artifact from the previous step.
* Process the data as needed, based on the findings in the Report (e.g., train/valid/test split, filtering, etc.).
* Create an Artifact with `type='dataset'`.
* Create a new data Table and add it to the new Artifact.

#### 3. Train a model
* Integrate W\&B with the ML framework being used.
* Store all hyperparameters in `wandb.config`.
* Download the dataset Artifact from the previous step.
* Initialize a Run with `job_type='model_training'`.
* Train a model.
* Create an Artifact with `type='model'`.
* Save final metrics to the `wandb.summary` dictionary.
* Iterate with a focus on optimizing a single key metric. Set minimum/maximum thresholds for the remaining metrics.

#### 4. Evaluate a model
* Initialize a Run with `job_type='model_evaluation'`.
* Download the model Artifact from the previous step.
* Ensure validation metrics are consistent with previous runs.
* Create and log appropriate tables and visualizations (prediction tables, confusion matrices, histograms, etc.).
* Create a Report with the results and perform error analysis.
* Once a model is selected for production, run inference on the test set and verify that metrics are realistic and similar to validation. If it passes evaluation, promote the model to the Model Registry. If not, it likely indicates validation overfitting or data leakage issues.
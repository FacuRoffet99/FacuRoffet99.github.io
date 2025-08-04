Title: Plataforma Weights & Biases
Date: 2025-08-04 12:15
Category: IA
Lang: es
Slug: ai_1
Author: Facundo Roffet

<!-- Hide default title -->
<style> h1.entry-title, h1.post-title, h1.title, h1:first-of-type {display: none;} </style>
<!-- Add custom title -->
<h2 style="text-align: center; font-size: 3em; color: rgba(12, 205, 76, 0.927);">Plataforma Weights & Biases</h2>

<!---------------------------------------------------------------------------->

> Estas son mis notas personales de los cursos [Weights & Biases 101](https://wandb.ai/site/courses/101/), [Weights & Biases 201: Model Registry](https://www.wandb.courses/courses/201-model-registry) y [Weights & Biases 101: Weave](https://wandb.ai/site/courses/weave/). 

<!---------------------------------------------------------------------------->

## Entrenar modelos de IA: W&B Models

#### Runs
* Una Run es una unidad de ejecución de código Python, que captura todo su contexto (versiones de librerías, hardware, métricas del sistema, estado de Git, etc).
* Puede incluir varios tipos de logs: métricas escalares, media (imágenes, histogramas, visualizaciones 3D, videos, audio), gráficos, tablas, etc.

#### Projects
* Un Project contine múltiples Runs y todos sus datos asociados.
* En la GUI, un Project tiene distintas pestañas:
    * Overview: información general.
    * Workspace: dashboard interactivo con paneles personalizables para visualizar resultados y comparar distintos Runs.
    * Runs: tabla que lista todas las Runs del Project.
    * Automations: procesos automáticos configurados.
    * Sweeps: conjunto de Runs que prueban automáticamente distintos hiperparámetros.
    * Reports: documentos flexibles e interactivos que se actualizan automáticamente, que sirve para compartir visualizaciones de un Project.

#### Artifacts
* Un Artifact permite rastrear y versionar cualquier dato que sea usado como entrada o salida de una Run: datasets, resultados de evaluación o checkpoints de modelos. Encapsula tanto archivos como carpetas, y se compone de:
    * Un nombre
    * Un tipo (los más útiles son 'dataset' y 'model')
    * Metadatos (en forma de diccionario)
    * Una descripción
    * Archivos, carpetas o referencias a almacenes externos de objetos (ej: s3)
* Cada vez que se registra un Artifact con el mismo nombre, es comparado con sus versiones anteriores. Si hay alguna diferencia, se crea una nueva versión indicada con un alias (v1, v2, v3).
* En la GUI de W&B se pueden ver datos de la versión, los metadatos, los archivos contenidos, el linaje (árbol genealógico con los procesos que crearon o usaron al Artifact) y un código de uso rápido.

#### Model Registry
* El Model Registry es un repositorio centralizado que alberga y organiza distintas Collections (también llamadas Model Tasks).
* Dicho de otra forma, en el Model Registry conviven todos los modelos aptos para salir a la luz, ya sea para ser testeados o puestos en producción.
* Una Collection se refiere a una tarea en particular (segmentación de edificios, clasificación de células, etc). Contiene Artifacts de tipo 'model' que fueron promovidos como candidatos a ser usados por equipos o procesos posteriores. Opinión personal: cada Project debería tener una única Collection asociada.
* Una Collection puede contener varias versiones de un mismo modelo, a las que se les pueden asignar distintos alias. Por ejemplo: staging y production.
* En el Model Registry se pueden establecer Automations, que son acciones automáticas a ejecutar cuando se añade un nuevo modelo o un nuevo alias. Por ejemplo: crear un Report.

<!---------------------------------------------------------------------------->

## Desarrollar aplicaciones con LLMs: W&B Weave

#### Calls
* Las Calls son el bloque fundamental de Weave, y representan una única ejecución de una función incluyendo sus entradas (argumentos), salidas (retornos) y metadatos (tiempos, excepciones, etc).
* Una Call puede estar relacionada a otra por medio de una relación de padre o de hijo, formando una estructura de árbol.
* Una vez creada, una Call no puede ser modificada. Lo único que puede hacerse es agregarle feedback o eliminarla (tanto por medio de código o de la GUI).

#### Traces
* Una Trace (o Span) es una colección de Calls que se encuentran en un mismo contexto de ejecución.

#### Objects (Models y Datasets) y Ops
* Un Object es un tipo de dato que Weave puede entender, serializar y versionar. Los Models y los Datasets son subclases de Objects; las Ops son funciones/métodos que actúan sobre Objects.
* Un Model sirve para versionar una app a lo largo del tiempo y entender cómo las modificaciones afectan a las respuestas. Están compuestos de una combinación de datos (configuración, checkpoint, etc) y códigos que definen como el modelo opera.
* Para crear un Model, se debe crear una clase que herede de `weave.Model` que contenga definiciones de tipo en todos sus campos y un método llamado `invoke` o `predict` con el decorator de Weave.
* Cuando se cambian los parámetros que definen a un Model, los cambios se loggean y la versión del modelo se actualiza automáticamente.
* Un Dataset es una colección de ejemplos para evaluar a un Model.

#### Evaluations
* Una Evaluation evalúa el desempeño de un Model en un Dataset usando una lista especificada de métricas (funciones o Scorers).
* La latencia y la cantidad de tokens son métricas que loggean por defecto.
* En la GUI se pueden hacer comparaciones exhaustivas entre distintas Evaluations.

<!---------------------------------------------------------------------------->

## Uso básico en código

#### Instalación
```
pip install wandb weave
```

#### Setup
```python
import wandb, weave
wandb.login()
@weave.op() # decorator for every function/method to track (automatically included in common LLM libraries)
```

#### Inicialización
```python
run = wandb.init(project=project_name, entity=entity_name, job_type=job_type, config=dict_with_configs)
client = weave.init(project_name) # only needed when using weave as standalone
```

#### Logs
Escalares y archivos:
```python
run.log({'train/epoch': epoch, 'train/train_loss': train_loss})
run.log({'val/val_loss': val_loss, 'val/accuracy': acc})
run.log_model(path_to_model_file) # same as creating an Artifact of type 'model'
```
Artifacts:
```python
# Log
artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
artifact.add_file(path_to_file)
artifact.add_dir(path_to_folder)
run.log_artifact(artifact)
# Retrieve
artifact = run.use_artifact(artifact_name:artifact_version)
artifact_dir = artifact.download()
```
Objects:
```python
# Log
weave.publish(variable_to_log, log_name)
# Retrieve
object = weave.ref(object_name:object_version).get()
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
# Calls
dataset = Dataset.from_calls(call_list)
# DataFrame
dataset = Dataset.from_pandas(df)
```

#### Evaluations
```python
evaluation = weave.Evaluation(dataset=dataset, scorers=scorers_list)
evaluation.evaluate(model)
```

#### Finalización
```python
# Only needed on notebooks
run.finish()
```
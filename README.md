# nntransfer  - _A simple framework for transfer learning experiments_

This framework provides all the tools necessary to quickly define a complex transfer experiment in a few lines of code, 
while being flexible enough to modify all components on every level.

## The Concept and Structure

This framework builds on top of [nnfabrik](https://github.com/sinzlab/nnfabrik), [neuralpredictors](https://github.com/sinzlab/neuralpredictors) and [datajoint](https://datajoint.io/). 
The code and conceptual structure involves several components that will be explained in the following:

### Configs
Config classes are designed to hold all settings that define a specific experiment. 
They allow default values to be set by assigning attributes in the init and easy overwrite ability by passing custom keyword arguments.
One key feature of these config objects is the option to access the hashed key that will be used in a datajoint schema.
Therefore it is easy to access and manipulate table entries for a given config object.

### Experiments

The configs in this framework are separated into `dataset`, `model` and `trainer` configs. 
Together these configs form an `Experiment`, which itself can be composed with other experiments in a `TransferExperiment`. 
`Experiment` and `TransferExperiment` objects encapsulate everything that needs to be run for a certain experiment.
An experiment could be an individual training run and a transfer experiment could be multiple experiments that hand over data or parameters chained together.

### Dataset
A dataset loader is supposed to gather a specific dataset (including all corresponding test sets), 
and prepare all data transformations as well as corresponding data loaders. 
The implementation can be found in the /dataset folder. 

### Model
The model-building functions can be found in the /models folder. 
Here we offer default implementations (with some adjustments) of some standard vision models. 

### Trainer
In order to train the defined model, we use the trainer function, that can be found in the /trainer folder. 
It is responsible for the whole training process including the batch iterations, loss computation, evaluation on the validation set and the logging of results per epoch and finally the final evaluation on the test sets.

### Mainloop-Modules
To allow most flexible usage of the default trainer function, we introduce main-loop modules. 
These modules can implement any of the following functions that will be called at their respective point in training: 
 - `pre_epoch`
   
 - `pre_forward` 

 - `post_forward`

 - `post_backward` 

 - `post_optimizer`

 - `post_epoch`

These functions should allow most common interactions with the training process, like an additional training objective for example.

## Recipes

Finally, to automatically execute a experiments (potentially in a distributed setting), simply define the concrete experiments in form of recipes, let our framework fill the corresponding tables and execute the training.
A template for this can be found here: [https://github.com/sinzlab/nntransfer_recipes](https://github.com/sinzlab/nntransfer_recipes)



## TFCNN - Tiny Fully Connected Neural Network Library

`26/10/20`
The content of this repository is the result of 22 days of work, prototyping, and research; TFCNN is a very efficient, simple, and fast neural network library to implement. Designed for binary classification.

`1/11/20`
The [`TFCNNv1_in8.h`](https://github.com/TFCNN/TFCNNv1/blob/main/TFCNNv1_int8.h) version is a quick modification to pack/quantise the Float32 operations down to int8's. Performance is hindered by casting operations, however the memory footprint is less.

### Version 2
- [`TFCNNv2.h`](https://github.com/TFCNN/TFCNNv2) is a Linux / UNIX / BSD version.

Version 2 is feature rich and probably superfluous, stick to [`TFCNNv1.h`](https://github.com/TFCNN/TFCNNv1/blob/main/TFCNNv1.h) if in doubt.

### Version 1
- `TFCNNv1.h` is a platform independent version.
```
int   createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units);
float processNetwork(network* net, const float* inputs, const learn_type learn);
void  resetNetwork(network* net);
void  destroyNetwork(network* net);
int   saveWeights(network* net, const char* file);
int   loadWeights(network* net, const char* file);

void setWeightInit(network* net, const weight_init_type u);
void setOptimiser(network* net, const optimiser u);
void setActivator(network* net, const activator u);
void setBatches(network* net, const uint u);
void setLearningRate(network* net, const float f);
void setGain(network* net, const float f);
void setDropout(network* net, const float f);
void setMomentum(network* net, const float f);
void setRMSAlpha(network* net, const float f);
void setTargetMin(network* net, const float f);
void setTargetMax(network* net, const float f);
void randomHyperparameters(network* net);

float qRandFloat(const float min, const float max);
float qRandWeight(const float min, const float max);
uint  qRand(const uint min, const uint umax);
```

#### ENUMS
```
enum 
{
    LEARN_MAX = 1,
    LEARN_MIN = 0,
    NO_LEARN  = -1
}
typedef learn_type;

enum 
{
    WEIGHT_INIT_UNIFORM             = 0,
    WEIGHT_INIT_UNIFORM_GLOROT      = 1,
    WEIGHT_INIT_UNIFORM_LECUN       = 2
}
typedef weight_init_type;

enum 
{
    IDENTITY = 0,
    ATAN     = 1,
    TANH     = 2,
    RELU     = 3,
    SIGMOID  = 4
}
typedef activator;

enum 
{
    OPTIM_SGD       = 0,
    OPTIM_MOMENTUM  = 1,
    OPTIM_NESTEROV  = 2,
    OPTIM_ADAGRAD   = 3,
    OPTIM_RMSPROP   = 4
}
typedef optimiser;
```

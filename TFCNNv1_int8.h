/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        October 2020 - TFCNNv1_int8
--------------------------------------------------
    Tiny Fully Connected Neural Network Library
    https://github.com/tfcnn

    This is a version that uses int8 quantisation
    to reduce the memory footprint.

    Skip down to SCALE_FACTOR if you desire
    to tune the quantisation min/max.
*/

#ifndef TFCNN_H
#define TFCNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define uint unsigned int
#define int8 signed char

/*
--------------------------------------
    structures
--------------------------------------
*/

// perceptron struct
struct
{
    int8* data;
    int8* momentum;
    int8 bias;
    int8 bias_momentum;
    uint weights;
}
typedef ptron;

// network struct
struct
{
    // hyperparameters
    uint  init;
    uint  activator;
    uint  optimiser;
    uint  batches;
    float rate;
    float gain;
    float dropout;
    float momentum;
    float rmsalpha;
    float min_target;
    float max_target;

    // layers
    ptron** layer;

    // count
    uint num_inputs;
    uint num_layers;
    uint num_layerunits;

    // backprop
    uint cbatches;
    float** output;
    float foutput;
    float error;
}
typedef network;

/*
--------------------------------------
    ERROR TYPES
--------------------------------------
*/

#define ERROR_UNINITIALISED_NETWORK -1
#define ERROR_TOOFEWINPUTS -2
#define ERROR_TOOFEWLAYERS -3
#define ERROR_TOOSMALL_LAYERSIZE -4
#define ERROR_ALLOC_LAYERS_ARRAY_FAIL -5
#define ERROR_ALLOC_LAYERS_FAIL -6
#define ERROR_ALLOC_OUTPUTLAYER_FAIL -7
#define ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL -8
#define ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL -9
#define ERROR_CREATE_FIRSTLAYER_FAIL -10
#define ERROR_CREATE_HIDDENLAYER_FAIL -11
#define ERROR_CREATE_OUTPUTLAYER_FAIL -12
#define ERROR_ALLOC_OUTPUT_ARRAY_FAIL -13
#define ERROR_ALLOC_OUTPUT_FAIL -14

/*
--------------------------------------
    DEFINES / ENUMS
--------------------------------------
*/

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
    WEIGHT_INIT_UNIFORM_LECUN       = 2,
    WEIGHT_INIT_NORMAL              = 3,
    WEIGHT_INIT_NORMAL_GLOROT       = 4,
    WEIGHT_INIT_NORMAL_LECUN        = 5
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

/*
--------------------------------------
    random functions
--------------------------------------
*/

#define FAST_PREDICTABLE_MODE

float qRandFloat(const float min, const float max);
float qRandWeight(const float min, const float max);
uint  qRand(const uint min, const uint umax);

/*
--------------------------------------
    accessors
--------------------------------------
*/

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

/*
--------------------------------------
    neural net functions
--------------------------------------
*/

int createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_hidden_layers, const uint num_layer_units);
float processNetwork(network* net, float* inputs, const learn_type learn);
void resetNetwork(network* net);
void destroyNetwork(network* net);
int saveWeights(network* net, const char* file);
int loadWeights(network* net, const char* file);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

float qRandFloat(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const float rv = (float)rand();
    if(rv == 0)
        return min;
    return ( (rv / (float)RAND_MAX) * (max-min) ) + min;
}

float qRandWeight(const float min, const float max)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / (float)RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

uint qRand(const uint min, const uint umax)
{
#ifndef FAST_PREDICTABLE_MODE
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
#endif
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / (float)RAND_MAX) * (max-min) ) + min;
}

/**********************************************/

#define SCALE_FACTOR 28  // Recommended factor: 32 - 128
                         // 8   = ~0.13 to 16.0
                         // 16  = ~0.07 to 8.0
                         // 32  = ~0.04 to 4.0
                         // 64  = ~0.02 to 2.0
                         // 128 = ~0.01 to 1.0
int8 packFloat(const float f)
{
    float r = (f * SCALE_FACTOR);
    if(r > 0){r += 0.5;}
    else if(r < 0){r -= 0.5;}
    if(r >= 127){return 127;}
    else if(r <= -128){return -128;}
    return (int8)r;
}
float unpackFloat(const int8 f)
{
    return (float)(f / SCALE_FACTOR);
}

/**********************************************/

static inline float atanDerivative(const float x) //atan()
{
    return 1 / (1 + (x*x));
}

static inline float tanhDerivative(const float x) //tanh()
{
    return 1 - (x*x);
}

/**********************************************/

static inline float relu(const float x)
{
    if(x < 0){return 0;}
    return x;
}

static inline float reluDerivative(const float x)
{
    if(x > 0)
        return 1;
    else
        return 0;
}

/**********************************************/

static inline float sigmoid(const float x)
{
    return 1 / (1 + exp(-x));
}

static inline float sigmoidDerivative(const float x)
{
    return x * (1 - x);
}

/**********************************************/

static inline float Derivative(const float x, const uint type)
{
    if(type == 1)
        return atanDerivative(x);
    else if(type == 2)
        return tanhDerivative(x);
    else if(type == 3)
        return reluDerivative(x);
    else if(type == 4)
        return sigmoidDerivative(x);
    
    return reluDerivative(x); // same as identity derivative
}

static inline float Activator(const float x, const uint type)
{
    if(type == 1)
        return atan(x);
    else if(type == 2)
        return tanh(x);
    else if(type == 3)
        return relu(x);
    else if(type == 4)
        return sigmoid(x);

    return x;
}

/**********************************************/

static inline float SGD(network* net, const float input, const float error)
{
    return net->rate * error * input;
}

static inline float Momentum(network* net, const float input, const float error, int8* momentum)
{
    // const float err = (_lrate * error * input);
    // const float ret = err + _lmomentum * momentum[0];
    // momentum[0] = err;
    // return ret;

    const float err = (net->rate * error * input) + net->momentum * unpackFloat(momentum[0]);
    momentum[0] = packFloat(err);
    return err;
}

static inline float Nesterov(network* net, const float input, const float error, int8* momentum)
{
    const float vp = unpackFloat(momentum[0]);
    const float v = net->momentum * vp + ( net->rate * error * input );
    const float n = v + net->momentum * (v - vp);
    momentum[0] = packFloat(v);
    return n;
}

static inline float ADAGrad(network* net, const float input, const float error, int8* momentum)
{
    const float err = error * input;
    const float momu = unpackFloat(momentum[0]);
    const float nmom = momu + (err * err);
    momentum[0] = packFloat(nmom);
    return (net->rate / sqrt(nmom + 1e-7)) * err;
}

static inline float RMSProp(network* net, const float input, const float error, int8* momentum)
{
    const float err = error * input;
    const float momu = unpackFloat(momentum[0]);
    const float nmom = net->rmsalpha * momu + (1 - net->rmsalpha) * (err * err);
    momentum[0] = packFloat(nmom);
    return (net->rate / sqrt(nmom + 1e-7)) * err;
}

static inline float Optimiser(network* net, const float input, const float error, int8* momentum)
{
    if(net->optimiser == 1)
        return Momentum(net, input, error, momentum);
    else if(net->optimiser == 2)
        return Nesterov(net, input, error, momentum);
    else if(net->optimiser == 3)
        return ADAGrad(net, input, error, momentum);
    else if(net->optimiser == 4)
        return RMSProp(net, input, error, momentum);
    
    return SGD(net, input, error);
}

/**********************************************/

static inline float doPerceptron(const float* in, ptron* p)
{
    float ro = 0;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * unpackFloat(p->data[i]);
    ro += p->bias;

    return ro;
}

/**********************************************/

int createPerceptron(ptron* p, const uint weights, const float d, const weight_init_type wit)
{
    p->data = malloc(weights * sizeof(int8));
    if(p->data == NULL)
        return ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL;

    p->momentum = malloc(weights * sizeof(int8));
    if(p->momentum == NULL)
        return ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL;

    p->weights = weights;

    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 3)
            p->data[i] = packFloat(qRandWeight(-d, d)); // uniform
        else
            p->data[i] = packFloat(qRandWeight(0, d));  // normal

        p->momentum[i] = 0;
    }

    p->bias = 0;
    p->bias_momentum = 0;

    return 0;
}

void resetPerceptron(ptron* p, const float d, const weight_init_type wit)
{
    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 3)
            p->data[i] = packFloat(qRandWeight(-d, d)); // uniform
        else
            p->data[i] = packFloat(qRandWeight(0, d));  // normal
        
        p->momentum[i] = 0;
    }

    p->bias = 0;
    p->bias_momentum = 0;
}

void setWeightInit(network* net, const weight_init_type u)
{
    if(net == NULL){return;}
    net->init = u;
}

void setOptimiser(network* net, const optimiser u)
{
    if(net == NULL){return;}
    net->optimiser = u;
}

void setActivator(network* net, const activator u)
{
    if(net == NULL){return;}
    net->activator = u;
}

void setBatches(network* net, const uint u)
{
    if(net == NULL){return;}
    net->batches = u;
}

void setLearningRate(network* net, const float f)
{
    if(net == NULL){return;}
    net->rate = f;
}

void setGain(network* net, const float f)
{
    if(net == NULL){return;}
    net->gain = f;
}

void setDropout(network* net, const float f)
{
    if(net == NULL){return;}
    net->dropout = f;
}

void setMomentum(network* net, const float f)
{
    if(net == NULL){return;}
    net->momentum = f;
}

void setRMSAlpha(network* net, const float f)
{
    if(net == NULL){return;}
    net->rmsalpha = f;
}

void setTargetMin(network* net, const float f)
{
    if(net == NULL){return;}
    net->min_target = f;
}

void setTargetMax(network* net, const float f)
{
    if(net == NULL){return;}
    net->max_target = f;
}

void randomHyperparameters(network* net)
{
    if(net == NULL){return;}
        
    net->init       = qRand(0, 5);
    net->activator  = qRand(1, 4);
    net->optimiser  = qRand(0, 4);
    net->rate       = qRandFloat(0.001, 0.03);
    net->dropout    = qRandFloat(0, 0.3);
    net->momentum   = qRandFloat(0.1, 0.9);
    net->rmsalpha   = qRandFloat(0.2, 0.99);
}

int createNetwork(network* net, const uint init_weights_type, const uint inputs, const uint hidden_layers, const uint layers_size)
{
    const uint layers = hidden_layers+2;

    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(inputs < 1)
        return ERROR_TOOFEWINPUTS;
    if(layers < 3)
        return ERROR_TOOFEWLAYERS;
    if(layers_size < 1)
        return ERROR_TOOSMALL_LAYERSIZE;

    // init net hyper parameters to some default
    net->num_layerunits = layers_size;
    net->num_inputs = inputs;
    net->num_layers = layers;
    net->init       = init_weights_type;
    net->activator  = 2;
    net->optimiser  = 1;
    net->batches    = 3;
    net->rate       = 0.01;
    net->gain       = 1.0;
    net->dropout    = 0.3;
    net->momentum   = 0.1;
    net->rmsalpha   = 0.2;
    net->min_target = 0;
    net->max_target = 1;
    net->cbatches   = 0;
    net->error      = 0;
    net->foutput    = 0;
    
    // create layers
    net->output = malloc((layers-1) * sizeof(float*));
    if(net->output == NULL)
        return ERROR_ALLOC_OUTPUT_ARRAY_FAIL;
    for(int i = 0; i < layers-1; i++)
    {
        net->output[i] = malloc(layers_size * sizeof(float));
        if(net->output[i] == NULL)
            return ERROR_ALLOC_OUTPUT_FAIL;
    }

    net->layer = malloc(layers * sizeof(ptron*));
    if(net->layer == NULL)
        return ERROR_ALLOC_LAYERS_ARRAY_FAIL;
    for(int i = 0; i < layers-1; i++)
    {
        net->layer[i] = malloc(layers_size * sizeof(ptron));
        if(net->layer[i] == NULL)
            return ERROR_ALLOC_LAYERS_FAIL;
    }

    net->layer[layers-1] = malloc(sizeof(ptron));
    if(net->layer[layers-1] == NULL)
        return ERROR_ALLOC_OUTPUTLAYER_FAIL;

    // init weight
    float d = 1; //WEIGHT_INIT_UNIFORM / WEIGHT_INIT_NORMAL
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/inputs);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/inputs);

    // create first layer perceptrons
    for(int i = 0; i < layers_size; i++)
        if(createPerceptron(&net->layer[0][i], inputs, d, net->init) < 0)
            return ERROR_CREATE_FIRSTLAYER_FAIL;
    
    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/layers_size);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/layers_size);

    // create hidden layers
    for(uint i = 1; i < layers-1; i++)
        for(int j = 0; j < layers_size; j++)
            if(createPerceptron(&net->layer[i][j], layers_size, d, net->init) < 0)
                return ERROR_CREATE_HIDDENLAYER_FAIL;

    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(layers_size+1));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(layers_size+1));

    // create output layer
    if(createPerceptron(&net->layer[layers-1][0], layers_size, d, net->init) < 0)
        return ERROR_CREATE_OUTPUTLAYER_FAIL;

    // done
    return 0;
}

float processNetwork(network* net, float* inputs, const int learn)
{
    // validate [it's ok, the output should be sigmoid 0-1 otherwise]
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
/**************************************
    Forward Prop
**************************************/

    // outputs per layer / unit
    float of[net->num_layers-1][net->num_layerunits];

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
        of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net->activator);

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net->activator);

    // binary classifier output layer
    const float output = sigmoid(doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]));

    // if it's just forward pass, return result.
    if(learn == NO_LEARN)
        return output;

/**************************************
    Backward Prop Error
**************************************/

    // reset accumulators if cbatches has been reset
    if(net->cbatches == 0)
    {
        for(int i = 0; i < net->num_layers-1; i++)
            memset(net->output[i], 0x00, net->num_layerunits * sizeof(float));

        net->foutput = 0;
        net->error = 0;
    }

    // batch accumulation of outputs
    net->foutput += output;
    for(int i = 0; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            net->output[i][j] += of[i][j];

    // accumulate output error
    float eo = net->max_target;
    if(learn == LEARN_MIN)
        eo = net->min_target;
    net->error += eo - output;

    // batching controller
    net->cbatches++;
    if(net->cbatches < net->batches)
    {
        return output;
    }
    else
    {
        // divide accumulators to mean
        net->error /= net->batches;
        net->foutput /= net->batches;

        for(int i = 0; i < net->num_layers-1; i++)
            for(int j = 0; j < net->num_layerunits; j++)
                net->output[i][j] /= net->batches;

        // reset batcher
        net->cbatches = 0;
    }

    // early return if error is 0
    if(net->error == 0)
        return output;

    // define error buffers
    float ef[net->num_layers-1][net->num_layerunits];

    // output (binary classifier) derivative error
    const float eout = net->gain * sigmoidDerivative(net->foutput) * net->error;

    // output derivative error layer before output layer
    float ler = 0;
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        ler += unpackFloat(net->layer[net->num_layers-1][0].data[j]) * eout;
    ler += unpackFloat(net->layer[net->num_layers-1][0].bias) * eout;
    for(int i = 0; i < net->num_layerunits; i++)
        ef[net->num_layers-2][i] = net->gain * Derivative(of[net->num_layers-2][i], net->activator) * ler;

    // output derivative error of all other layers
    for(int i = net->num_layers-3; i >= 0; i--)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            float ler = 0;
            for(int k = 0; k < net->layer[i+1][j].weights; k++)
                ler += unpackFloat(net->layer[i+1][j].data[k]) * ef[i+1][j];
            ler += unpackFloat(net->layer[i+1][j].bias) * ef[i+1][j];

            ef[i][j] = net->gain * Derivative(of[i][j], net->activator) * ler;
        }
    }

/**************************************
    Update Weights
**************************************/
    
    // update input layer weights
    for(int j = 0; j < net->num_layerunits; j++)
    {
        if(net->dropout != 0 && qRandFloat(0, 1) <= net->dropout)
            continue;

        for(int k = 0; k < net->layer[0][j].weights; k++)
            net->layer[0][j].data[k] = packFloat(net->layer[0][j].data[k] + Optimiser(net, inputs[k], ef[0][j], &net->layer[0][j].momentum[k]));

        net->layer[0][j].bias = packFloat(net->layer[0][j].bias + Optimiser(net, 1, ef[0][j], &net->layer[0][j].bias_momentum));
    }

    // update hidden layer weights
    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            if(net->dropout != 0 && qRandFloat(0, 1) <= net->dropout)
                continue;

            for(int k = 0; k < net->layer[i][j].weights; k++)
                net->layer[i][j].data[k] = packFloat(net->layer[i][j].data[k] + Optimiser(net, net->output[i-1][j], ef[i][j], &net->layer[i][j].momentum[k]));

            net->layer[i][j].bias = packFloat(net->layer[i][j].bias + Optimiser(net, 1, ef[i][j], &net->layer[i][j].bias_momentum));
        }
    }

    // update output layer weights
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        net->layer[net->num_layers-1][0].data[j] = packFloat(net->layer[net->num_layers-1][0].data[j] + Optimiser(net, net->output[net->num_layers-2][j], eout, &net->layer[net->num_layers-1][0].momentum[j]));

    net->layer[net->num_layers-1][0].bias = packFloat(net->layer[net->num_layers-1][0].bias + Optimiser(net, 1, eout, &net->layer[net->num_layers-1][0].bias_momentum));

    // done, return forward prop output
    return output;
}

void resetNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // reset batching counter
    for(int i = 0; i < net->num_layers-1; i++)
        memset(net->output[i], 0x00, net->num_layerunits * sizeof(float));
    net->cbatches = 0;
    net->foutput = 0;
    net->error = 0;
    
    // init weight
    float d = 1; //WEIGHT_INIT_RANDOM
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/net->num_inputs);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/net->num_inputs);

    // reset first layer perceptrons
    for(int i = 0; i < net->num_layerunits; i++)
        resetPerceptron(&net->layer[0][i], d, net->init);
    
    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrt(3.0/net->num_layerunits);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrt(1.0/net->num_layerunits);

    // reset hidden layers
    for(uint i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            resetPerceptron(&net->layer[i][j], d, net->init);

    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrt(6.0/(net->num_layerunits+1));
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrt(2.0/(net->num_layerunits+1));

    // reset output layer
    resetPerceptron(&net->layer[net->num_layers-1][0], d, net->init);
}

void destroyNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // free all perceptron data, percepron units and layers
    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            free(net->layer[i][j].data);
            free(net->layer[i][j].momentum);
        }
        free(net->layer[i]);
    }
    free(net->layer[net->num_layers-1][0].data);
    free(net->layer[net->num_layers-1][0].momentum);
    free(net->layer[net->num_layers-1]);
    free(net->layer);

    // free output buffers
    for(int i = 0; i < net->num_layers-1; i++)
        free(net->output[i]);
    free(net->output);
}

int saveWeights(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;

    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        for(int i = 0; i < net->num_layers-1; i++)
        {
            for(int j = 0; j < net->num_layerunits; j++)
            {
                if(fwrite(&net->layer[i][j].data[0], 1, net->layer[i][j].weights*sizeof(int8), f) != net->layer[i][j].weights*sizeof(int8))
                    return -1;
                
                if(fwrite(&net->layer[i][j].momentum[0], 1, net->layer[i][j].weights*sizeof(int8), f) != net->layer[i][j].weights*sizeof(int8))
                    return -1;

                if(fwrite(&net->layer[i][j].bias, 1, sizeof(int8), f) != sizeof(int8))
                    return -1;
                
                if(fwrite(&net->layer[i][j].bias_momentum, 1, sizeof(int8), f) != sizeof(int8))
                    return -1;
            }
        }

        if(fwrite(&net->layer[net->num_layers-1][0].data[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(int8), f) != net->layer[net->num_layers-1][0].weights*sizeof(int8))
            return -1;
        
        if(fwrite(&net->layer[net->num_layers-1][0].momentum[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(int8), f) != net->layer[net->num_layers-1][0].weights*sizeof(int8))
            return -1;

        if(fwrite(&net->layer[net->num_layers-1][0].bias, 1, sizeof(int8), f) != sizeof(int8))
            return -1;
        
        if(fwrite(&net->layer[net->num_layers-1][0].bias_momentum, 1, sizeof(int8), f) != sizeof(int8))
            return -1;

        fclose(f);
    }

    return 0;
}

int loadWeights(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -1;

    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            if(fread(&net->layer[i][j].data[0], 1, net->layer[i][j].weights*sizeof(int8), f) != net->layer[i][j].weights*sizeof(int8))
                return -1;

            if(fread(&net->layer[i][j].momentum[0], 1, net->layer[i][j].weights*sizeof(int8), f) != net->layer[i][j].weights*sizeof(int8))
                return -1;

            if(fread(&net->layer[i][j].bias, 1, sizeof(int8), f) != sizeof(int8))
                return -1;

            if(fread(&net->layer[i][j].bias_momentum, 1, sizeof(int8), f) != sizeof(int8))
                return -1;
        }
    }

    if(fread(&net->layer[net->num_layers-1][0].data[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(int8), f) != net->layer[net->num_layers-1][0].weights*sizeof(int8))
        return -1;

    if(fread(&net->layer[net->num_layers-1][0].momentum[0], 1, net->layer[net->num_layers-1][0].weights*sizeof(int8), f) != net->layer[net->num_layers-1][0].weights*sizeof(int8))
        return -1;

    if(fread(&net->layer[net->num_layers-1][0].bias, 1, sizeof(int8), f) != sizeof(int8))
        return -1;

    if(fread(&net->layer[net->num_layers-1][0].bias_momentum, 1, sizeof(int8), f) != sizeof(int8))
        return -1;

    fclose(f);
    return 0;
}

#endif


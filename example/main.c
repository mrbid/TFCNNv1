/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
        October 2020 - TFCNNv1 - Example Demo
--------------------------------------------------
    Tiny Fully Connected Neural Network Library
    https://github.com/tfcnn
*/

#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wformat-zero-length"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <locale.h>
#include <sys/file.h>

#include "TFCNNv1.h"

///

#define TABLE_SIZE_MAX 10000
#define DIGEST_SIZE 8
#define WORD_SIZE 32
#define MESSAGE_SIZE WORD_SIZE*DIGEST_SIZE

#define DATA_TRAIN_PERCENT 0.7
#define DATA_SIZE 995
#define OUTPUT_QUOTES 33333
#define HIDDEN_SIZE 256
#define HIDDEN_LAYERS 2
#define TRAINING_LOOPS 1

///

uint _log = 0;

// discriminator 
network net;

// normalised training data
float digest[DATA_SIZE][DIGEST_SIZE] = {0};

//word lookup table / index
char wtable[TABLE_SIZE_MAX][WORD_SIZE] = {0};
uint TABLE_SIZE = 0;
uint TABLE_SIZE_H = 0;


//*************************************
// utility functions
//*************************************

void newSRAND()
{
    srand(time(0));
}

void loadTable(const char* file)
{
    FILE* f = fopen(file, "r");
    if(f)
    {
        uint index = 0;
        while(fgets(wtable[index], WORD_SIZE, f) != NULL)
        {
            char* pos = strchr(wtable[index], '\n');
            if(pos != NULL)
                *pos = '\0';
            
            index++;
            if(index == TABLE_SIZE_MAX)
                break;
        }
        TABLE_SIZE = index;
        TABLE_SIZE_H = TABLE_SIZE / 2;
        fclose(f);
    }
}

float getWordNorm(const char* word)
{
    for(uint i = 0; i < TABLE_SIZE; i++)
        if(strcmp(word, wtable[i]) == 0)
            return (((double)i) / (double)(TABLE_SIZE_H))-1.0;

    return 0;
}

//https://stackoverflow.com/questions/30432856/best-way-to-get-number-of-lines-in-a-file-c
uint countLines(const char* file)
{
    uint lines = 0;
    FILE *fp = fopen(file, "r");
    if(fp != NULL)
    {
        while(EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
            ++lines;
        
        fclose(fp);
    }
    return lines;
}

void clearFile(const char* file)
{
    FILE *f = fopen(file, "w");
    if(f != NULL)
    {
        fprintf(f, "");
        fclose(f);
    }
}

void timestamp()
{
    const time_t ltime = time(0);
    printf("%s", asctime(localtime(&ltime)));
}

float rmseDiscriminator(const uint start, const uint end)
{
    float squaremean = 0;
    for(uint i = start; i < end; i++)
    {
        const float r = 1 - processNetwork(&net, &digest[i][0], NO_LEARN);
        squaremean += r*r;
    }
    squaremean /= DATA_SIZE;
    return sqrt(squaremean);
}

void loadDataset(const char* file)
{
    const time_t st = time(0);

    // read training data [every input is truncated to 256 characters]
    FILE* f = fopen(file, "r");
    if(f)
    {
        char line[MESSAGE_SIZE];
        uint index = 0;
        while(fgets(line, MESSAGE_SIZE, f) != NULL)
        {
            char* pos = strchr(line, '\n');
            if(pos != NULL)
                *pos = '\0';
            uint i = 0;
            char* w = strtok(line, " ");
            
            while(w != NULL)
            {
                digest[index][i] = getWordNorm(w); //normalise
                w = strtok(NULL, " ");
                i++;
            }

            index++;
            if(index == DATA_SIZE)
                break;
        }
        fclose(f);
    }

    printf("Training Data Loaded.\n");
    printf("Time Taken: %.2f mins\n\n", ((double)(time(0)-st)) / 60.0);
}

float trainDataset(const uint start, const uint end)
{
    float rmse = 0;

    // train discriminator
    const time_t st = time(0);
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = start; i < end; i++)
        {
            // train discriminator on data
            processNetwork(&net, &digest[i][0], LEARN_MAX);

            // detrain discriminator on random word sequences 
            float output[DIGEST_SIZE] = {0};
            const int len = qRand(1, DIGEST_SIZE);
            for(int i = 0; i < len; i++)
                output[i] = (((double)qRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0; //qRandWeight(-1, 1);
            processNetwork(&net, &output[0], LEARN_MIN);

            if(_log == 1)
                printf("Training Iteration (%u / %u) [%u / %u]\n RAND | REAL\n", i+1, DATA_SIZE, j+1, TRAINING_LOOPS);

            if(_log == 1)
            {
                for(int k = 0; k < DIGEST_SIZE; k++)
                    printf("%+.2f : %+.2f\n", output[k], digest[i][k]);

                printf("\n");
            }
        }

        rmse = rmseDiscriminator(DATA_SIZE * DATA_TRAIN_PERCENT, DATA_SIZE);
        if(_log == 1)
            printf("RMSE: %f :: %lus\n", rmse, time(0)-st);
        if(_log == 2)
            printf("RMSE:          %.2f :: %lus\n", rmse, time(0)-st);
    }

    // return rmse
    return rmse;
}


//*************************************
// program functions
//*************************************

void consoleAsk()
{
    // what percentage human is this ?
    while(1)
    {
        char str[MESSAGE_SIZE] = {0};
        float nstr[DIGEST_SIZE] = {0};
        printf(": ");
        fgets(str, MESSAGE_SIZE, stdin);
        str[strlen(str)-1] = 0x00; //remove '\n'

        //normalise words
        uint i = 0;
        char* w = strtok(str, " ");
        while(w != NULL)
        {
            nstr[i] = getWordNorm(w);
            w = strtok(NULL, " ");
            i++;
        }

        const float r = processNetwork(&net, nstr, NO_LEARN);
        printf("This is %.2f%% (%.2f) Human.\n", r * 100, r);
    }
}

float isHuman(char* str)
{
    float nstr[DIGEST_SIZE] = {0};

    //normalise words
    uint i = 0;
    char* w = strtok(str, " ");
    while(w != NULL)
    {
        nstr[i] = getWordNorm(w);
        printf("> %s : %f\n", w, nstr[i]);
        w = strtok(NULL, " ");
        i++;
    }

    const float r = processNetwork(&net, nstr, NO_LEARN);
    return r*100;
}

float rndScentence(const uint silent)
{
    float nstr[DIGEST_SIZE] = {0};
    const int len = qRand(1, DIGEST_SIZE);
    for(int i = 0; i < len; i++)
        nstr[i] = (((double)qRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

    for(int i = 0; i < DIGEST_SIZE; i++)
    {
        const uint ind = (((double)nstr[i]+1.0)*(double)TABLE_SIZE_H)+0.5;
        if(nstr[i] != 0 && silent == 0)
            printf("%s (%.2f) ", wtable[ind], nstr[i]);
    }

    if(silent == 0)
        printf("\n");

    const float r = processNetwork(&net, nstr, NO_LEARN);
    return r*100;
}

uint rndGen(const char* file, const float max, const uint timeout)
{
    FILE* f = fopen(file, "w");
    if(f != NULL)
    {
        uint count = 0;
        time_t st = time(0);
        for(int k = 0; k < OUTPUT_QUOTES; NULL)
        {
            float nstr[DIGEST_SIZE] = {0};
            const int len = qRand(1, DIGEST_SIZE);
            for(int i = 0; i < len; i++)
                nstr[i] = (((double)qRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

            const float r = processNetwork(&net, nstr, NO_LEARN);
            if(1-r < max)
            {
                for(int i = 0; i < DIGEST_SIZE; i++)
                {
                    const uint ind = (((double)nstr[i]+1.0)*(double)TABLE_SIZE_H)+0.5;
                    if(nstr[i] != 0)
                    {
                        fprintf(f, "%s ", wtable[ind]);
                        if(_log == 1)
                            printf("%s ", wtable[ind]);
                    }
                }
                
                k++;
                count++;
                fprintf(f, "\n");
                if(_log == 1)
                    printf("\n");
            }

            if(timeout == 1 && time(0) - st > 9) // after 9 seconds
            {
                if(count < 450)
                {
                    printf(":: Terminated at a RPS of %u/50 per second.\n", count/9);
                    return 0; // if the output rate was less than 50 per second, just quit.
                }
                
                count = 0;
                st = time(0);
            }
        }

        fclose(f);
    }

    return 1;
}

float hasFailed(const uint resolution)
{
    int failvariance = 0;
    for(int i = 0; i < 100*resolution; i++)
    {
        const float r = rndScentence(1);
        if(r < 50)
            failvariance++;
    }
    if(resolution == 1)
        return failvariance;
    else
        return (double)failvariance / (double)resolution;
}

uint huntBestWeights(float* rmse)
{
    *rmse = 0;
    float fv = 0;
    float min = 70;
    const float max = 96.0;
    float highest = 0;
    time_t st = time(0);
    while(fv < min || fv > max) //we want random string to fail at-least 70% of the time / but we don't want it to fail all of the time
    {
        newSRAND(); //kill any predictability in the random generator

        randomHyperparameters(&net);
        resetNetwork(&net);

        *rmse = trainDataset(0, DATA_SIZE * DATA_TRAIN_PERCENT);

        fv = hasFailed(100);
        if(fv <= max && fv > highest)
            highest = fv;

        if(time(0) - st > 540) //If taking longer than 3 mins just settle with the highest logged in that period
        {
            min = highest;
            highest = 0;
            st = time(0);
            printf("Taking too long, new target: %.2f\n", min);
        }

        printf("RMSE: %f / Fail: %.2f\n", *rmse, fv);
    }
    return fv; // fail variance
}

void rndBest()
{
    _log = 2;
    loadDataset("botmsg.txt");

    // load the last lowest fv target
    FILE* f = fopen("gs.dat", "r");
    while(f == NULL)
    {
        f = fopen("gs.dat", "r");
        usleep(1000); //1ms
    }
    float min = 0;
    while(fread(&min, 1, sizeof(float), f) != sizeof(float))
        usleep(1000);
    fclose(f);
    printf("Start fail variance: %.2f\n\n", min);

    // find a new lowest fv target
    while(1)
    {
        float rmse = 0;
        const time_t st = time(0);
        float fv = 0;
        const float max = 96.0;
        while(fv < min || fv > max) //we want random string to fail at-least some percent of the time more than 50% preferably
        {
            newSRAND(); //kill any predictability in the random generator

            randomHyperparameters(&net);

            printf("Weight Init:   %u\n", net.init);
            printf("Activator:     %u\n", net.activator);
            printf("Optimiser:     %u\n", net.optimiser);
            printf("Learning Rate: %f\n", net.rate);
            printf("Dropout:       %f\n", net.dropout);
            if(net.optimiser == OPTIM_MOMENTUM || net.optimiser == OPTIM_NESTEROV)
                printf("Momentum:      %f\n", net.momentum);
            else if(net.optimiser == OPTIM_RMSPROP)
                printf("RMSProp Alpha: %f\n", net.rmsalpha);

            printf("~\n");

            resetNetwork(&net);
            rmse = trainDataset(0, DATA_SIZE * DATA_TRAIN_PERCENT);
            
            const time_t st2 = time(0);
            fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
        }

        // this allows multiple processes to compete on the best weights
        f = fopen("gs.dat", "r+");
        while(f == NULL)
        {
            f = fopen("gs.dat", "r+");
            usleep(1000); //1ms
        }
        while(fread(&min, 1, sizeof(float), f) != sizeof(float))
            usleep(1000);
        if(min < fv)
        {
            while(flock(fileno(f), LOCK_EX) == -1)
                usleep(1000);
            while(fseek(f, 0, SEEK_SET) < 0)
                usleep(1000);
            while(fwrite(&fv, 1, sizeof(float), f) != sizeof(float))
                usleep(1000);
            flock(fileno(f), LOCK_UN);

            min = fv;

            saveWeights(&net, "weights.dat");
        }
        fclose(f);

        // done    
        const double time_taken = ((double)(time(0)-st)) / 60.0;
        printf("Time Taken: %.2f mins\n\n", time_taken);

        if(fv >= 99.0 || min >= max)
            exit(0);
    }
    exit(0);
}

void bestSetting(const float min)
{
    _log = 2;
    loadDataset("botmsg.txt");

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0, a8=0;
    uint count=0, c1=0, c2=0, c3=0, c4=0, c5=0;

    uint oc[5] = {0};
    uint ic[8] = {0};
    uint ac[5] = {0};

    // find a new lowest fv target
    while(1)
    {
        float rmse = 0;
        const time_t st = time(0);
        float fv = 0;
        const float max = 96.0;
        while(fv < min || fv > max) //we want random string to fail at-least some percent of the time more than 50% preferably
        {
            newSRAND(); //kill any predictability in the random generator

            randomHyperparameters(&net);
            resetNetwork(&net);

            rmse = trainDataset(0, DATA_SIZE * DATA_TRAIN_PERCENT);

            const time_t st2 = time(0);
            fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
        }

        if(rmse <= 0)
        {
            c5++;
            continue;
        }

        a0 += fv;
        a1 += rmse;
        a3 += net.rate;
        a4 += net.dropout;
        count++;
        oc[net.optimiser]++;
        ic[net.init]++;
        ac[net.activator]++;

        if(net.optimiser == OPTIM_MOMENTUM || net.optimiser == OPTIM_NESTEROV)
        {
            a5 += net.momentum;
            c1++;
        }
        else if(net.optimiser == OPTIM_RMSPROP)
        {
            a6 += net.rmsalpha;
            c2++;
        }

        // keep a log of the average
        FILE* f = fopen("best_average.txt", "w");
        if(f != NULL)
        {
            while(flock(fileno(f), LOCK_EX) == -1)
                usleep(1000);

            fprintf(f, "Iterations: %u\n", count);
            fprintf(f, "Fail Variance: %f\n", a0/count);
            fprintf(f, "RMSE: %f\n", a1/count);
            if(c5 != 0)
                fprintf(f, "ZERO-RMSE: %u\n", c5);
            fprintf(f, "L-Rate: %f\n", a3/count);
            fprintf(f, "Dropout: %f\n", a4/count);
            if(a5 != 0)
                fprintf(f, "Momentum: %f / %u\n", a5/c1, c1);
            if(a6 != 0)
                fprintf(f, "RMS Alpha: %f / %u\n", a6/c2, c2);
            fprintf(f, "\n");
            for(uint i = 0; i < 5; i++)
                fprintf(f, "Optimiser-%u: %u\n", i, oc[i]);
            for(uint i = 0; i < 8; i++)
                fprintf(f, "Weight Init-%u: %u\n", i, ic[i]);
            for(uint i = 0; i < 5; i++)
                fprintf(f, "Activator-%u: %u\n", i, ac[i]);
            fprintf(f, "\n");

            flock(fileno(f), LOCK_UN);
            fclose(f);
        }

        // done    
        const double time_taken = ((double)(time(0)-st)) / 60.0;
        printf("\nbest_average.txt updated ; Time Taken: %.2f mins\n\n", time_taken);
    }
    exit(0);
}

void resetState(const float min)
{
    remove("weights.dat");
            
    FILE* f = fopen("gs.dat", "w");
    while(f == NULL)
    {
        f = fopen("gs.dat", "w");
        usleep(1000);
    }
    while(fwrite(&min, 1, sizeof(float), f) != sizeof(float))
        usleep(1000);
    fclose(f);
}

void saveFailVariance(const float fv)
{
    FILE* f = fopen("gs.dat", "w");
    while(f == NULL)
    {
        f = fopen("gs.dat", "w");
        usleep(1000); //1ms
    }
    while(flock(fileno(f), LOCK_EX) == -1)
        usleep(1000);
    while(fseek(f, 0, SEEK_SET) < 0)
        usleep(1000);
    while(fwrite(&fv, 1, sizeof(float), f) != sizeof(float))
        usleep(1000);
    flock(fileno(f), LOCK_UN);
    fclose(f);
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init discriminator
    const int ret = createNetwork(&net, WEIGHT_INIT_UNIFORM_LECUN, DIGEST_SIZE, HIDDEN_LAYERS, HIDDEN_SIZE);
    if(ret < 0)
    {
        printf("createNetwork() failed: %i\n", ret);
        exit(0);
    }

    // load lookup table
    loadTable("botdict.txt");

    // are we issuing any commands?
    if(argc == 3)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            resetState(70);
            loadDataset(argv[2]);
            trainDataset(0, DATA_SIZE);
            const time_t st2 = time(0);
            const float fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
            saveWeights(&net, "weights.dat");
            saveFailVariance(fv);
            exit(0);
        }

        if(strcmp(argv[1], "bestset") == 0)
            bestSetting(atoi(argv[2]));

        if(strcmp(argv[1], "gen") == 0)
        {
            _log = 1;
            printf("Brute forcing with an error of: %s\n\n", argv[2]);
            loadWeights(&net, "weights.dat");
            rndGen("out.txt", atof(argv[2]), 0);
            exit(0);
        }

        if(strcmp(argv[1], "reset") == 0)
        {
            resetState(atoi(argv[2]));
            printf("Weights and multi-process descriptor reset.\n");
            exit(0);
        }
    }

    if(argc == 2)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            resetState(70);

            loadDataset("botmsg.txt");
            trainDataset(0, DATA_SIZE);
            const time_t st2 = time(0);
            const float fv = hasFailed(100);
            printf("Fail Variance: %.2f :: %lus\n---------------\n", fv, time(0)-st2);
            saveWeights(&net, "weights.dat");
            saveFailVariance(fv);

            srand(net.num_layerunits+net.num_inputs*net.num_layers+net.init+net.activator*net.optimiser);
            const uint l = qRand(0, net.num_layers-2);
            const uint lu = qRand(0, net.num_layerunits-1);
            const uint a = qRand(0, net.layer[l][lu].weights-1);
            const uint b = qRand(0, net.layer[l][lu].weights-1);
            const uint c = qRand(0, net.layer[l][lu].weights-1);
            printf("RWV: [%u][%u]\n", l, lu);
            printf("RWV[%u]: %i\n", a, net.layer[l][lu].data[a]);
            printf("RWV[%u]: %i\n", b, net.layer[l][lu].data[b]);
            printf("RWV[%u]: %i\n", c, net.layer[l][lu].data[c]);

            exit(0);
        }

        if(strcmp(argv[1], "bestset") == 0)
            bestSetting(70);

        if(strcmp(argv[1], "best") == 0)
            rndBest();
        
        if(strcmp(argv[1], "reset") == 0)
        {
            resetState(70);
            printf("Weights and multi-process descriptor reset.\n");
            exit(0);
        }

        if(strcmp(argv[1], "check") == 0)
        {
            FILE* f = fopen("gs.dat", "r");
            while(f == NULL)
            {
                f = fopen("gs.dat", "r");
                usleep(1000); //1ms
            }
            float fv = 0;
            while(fread(&fv, 1, sizeof(float), f) != sizeof(float))
                usleep(1000);
            fclose(f);

            printf("Current weights have a fail variance of %f.\n", fv);

            struct stat st;
            const int sr = stat("weights.dat", &st);
            setlocale(LC_NUMERIC, "");
            if(sr == 0 && st.st_size > 0)
                printf("%'.0f kb / %'.2f mb / %'.2f gb\n", (double)st.st_size / 1000, ((((double)st.st_size) / 1000) / 1000), ((((double)st.st_size) / 1000) / 1000) / 1000);
            else
                printf("weights.dat is 0 bytes.\n");

            loadWeights(&net, "weights.dat");

            printf("\nLoaded Settings:\n");
            printf("num_neurons: %u\n", net.num_layerunits);
            printf("num_inputs:  %u\n", net.num_inputs);
            printf("num_layers:  %u\n", net.num_layers);
            printf("weight_init: %u\n", net.init);
            printf("activator:   %u\n", net.activator);
            printf("optimiser:   %u\n", net.optimiser);
            printf("batches:     %u\n", net.batches);
            printf("learn-rate:  %f\n", net.rate);
            printf("learn-gain:  %f\n", net.gain);
            printf("dropout:     %f\n", net.dropout);
            printf("momentum:    %f\n", net.momentum);
            printf("rmsalpha:    %f\n", net.rmsalpha);
            printf("min_target:  %f\n", net.min_target);
            printf("max_target:  %f\n", net.max_target);

            // random weight selection for weight load/save verification
            srand(net.num_layerunits+net.num_inputs*net.num_layers+net.init+net.activator*net.optimiser);
            const uint l = qRand(0, net.num_layers-2);
            const uint lu = qRand(0, net.num_layerunits-1);
            const uint a = qRand(0, net.layer[l][lu].weights-1);
            const uint b = qRand(0, net.layer[l][lu].weights-1);
            const uint c = qRand(0, net.layer[l][lu].weights-1);
            printf("RWV: [%u][%u]\n", l, lu);
            printf("RWV[%u]: %i\n", a, net.layer[l][lu].data[a]);
            printf("RWV[%u]: %i\n", b, net.layer[l][lu].data[b]);
            printf("RWV[%u]: %i\n", c, net.layer[l][lu].data[c]);

            exit(0);
        }

        ///////////////////////////
        loadWeights(&net, "weights.dat");
        ///////////////////////////

        if(strcmp(argv[1], "ask") == 0)
            consoleAsk();

        if(strcmp(argv[1], "rnd") == 0)
        {
            newSRAND();
            printf("> %.2f\n", rndScentence(0));
            exit(0);
        }

        if(strcmp(argv[1], "gen") == 0)
        {
            _log = 1;
            rndGen("out.txt", 0.5, 0);
            exit(0);
        }

        if(strcmp(argv[1], "rndloop") == 0)
        {
            newSRAND();
            while(1)
                printf("> %.2f\n\n", rndScentence(0));
        }

        char in[MESSAGE_SIZE] = {0};
        snprintf(in, MESSAGE_SIZE, "%s", argv[1]);
        printf("%.2f\n", isHuman(in));
        exit(0);
    }

    // no commands ? run as service
    loadWeights(&net, "weights.dat");

    // main loop
    printf("Running ! ...\n\n");
    while(1)
    {
        if(countLines("botmsg.txt") >= DATA_SIZE)
        {
            timestamp();
            const time_t st = time(0);
            memset(&wtable, 0x00, TABLE_SIZE_MAX*WORD_SIZE);
            resetState(70);
            loadTable("botdict.txt");
            loadDataset("botmsg.txt");
            clearFile("botmsg.txt");

            float rmse = 0;
            uint fv = huntBestWeights(&rmse);
            while(rndGen("out.txt", 0.2, 1) == 0)
                fv = huntBestWeights(&rmse);

            saveWeights(&net, "weights.dat");
            printf("Just generated a new dataset.\n");
            timestamp();
            const double time_taken = ((double)(time(0)-st)) / 60.0;
            printf("Time Taken: %.2f mins\n\n", time_taken);

            FILE* f = fopen("portstat.txt", "w");
            if(f != NULL)
            {
                const time_t ltime = time(0);
                setlocale(LC_NUMERIC, "");
                fprintf(f, "Trained with an RMSE of %f and Fail Variance of %u (higher is better) on;\n%sTime Taken: %.2f minutes\nDigest size: %'u\n", rmse, fv, asctime(localtime(&ltime)), time_taken, DATA_SIZE);
                fprintf(f, "Activator: %u\n", net.activator);
                fprintf(f, "Weight Init: %u\n", net.init);
                fprintf(f, "L-Rate: %f\n", net.rate);
                fprintf(f, "Dropout: %f\n", net.dropout);
                if(net.optimiser == OPTIM_MOMENTUM || net.optimiser == OPTIM_NESTEROV)
                    fprintf(f, "Momentum: %f\n", net.momentum);
                else if(net.optimiser == OPTIM_RMSPROP)
                    fprintf(f, "RMS Alpha: %f\n", net.rmsalpha);
                fprintf(f, "Optimiser: %u\n\n", net.optimiser);
                fprintf(f, "I have %'u neurons with %'u configurable weights.\n", net.num_layers * net.num_layerunits, (net.num_layers-2) * net.num_layerunits * net.num_layerunits + net.num_layerunits * net.num_inputs + net.num_layerunits+1);
                fclose(f);
            }
        }

        sleep(9);
    }

    // done
    return 0;
}


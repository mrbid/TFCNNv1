This example is a fully working parrot bot service which makes use of `TFCNNv1.h`, it will randomly generate quotes that score highly against the binary classifier / discriminator.

It can be accessed via command line or run as a service where scripts aggregate data into its `botmsg.txt` and `botdict.txt` files, the former being for full string sentences and the latter should be a list of every unique word in every sentence included in `botmsg.txt`.

A php script is enclosed which will aggregate data from a telegram bot that has privacy mode disabled.

Compile using `gcc main.c -lm -Ofast -o bot` for example, ensure `TFCNNv1.h` is included in the same directory.

## Example Usage - Commandline Arguments
- ```./bot retrain <optional file path>```
<br>Train the network from the provided dataset.

- ```./bot check```
<br>Chech the fail variance of the current weights.

- ```./bot reset <optional fail variance lower limit>```
<br>Reset the current weights. The optional parameter allows you to set the minimum viable fail variance value for a set of computed weights, all weights below this value are discarded.

- ```./bot best```
<br>Randomly iterate each parameter and recompute the weights until the best solution is found. This function is multi-process safe.

- ```./bot bestset <optional fail variance lower limit>```
<br>Randomly iterate each parameter and recompute the weights outputting the average best parameters to `best_average.txt`. This function is multi-process safe.

- ```./bot "this is an example scentence"```
<br>Get a percentage of likelyhood that the sampled dataset wrote the provided message.

- ```./bot rnd```
<br>Get the percentage of likelyhood that the sampled dataset wrote a provided random message.

- ```./bot ask```
<br>A never ending console loop where you get to ask what percentage likelyhood the sampled dataset wrote a given message.

- ```./bot gen <optional max error>```
<br>The brute-force random message/quote generator.

- ```./bot```
<br>Bot service, will digest the botmsg.txt and botdict.txt every x messages and generate a new set of quotes.

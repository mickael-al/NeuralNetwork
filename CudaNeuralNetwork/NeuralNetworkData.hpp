#ifndef __NEURAL_NETWORK_DATA__
#define __NEURAL_NETWORK_DATA__

struct NeuralNetworkData
{
	int nb_input_layer;
	int nb_output_layer;
	int nb_hiden_layer;
	int nb_col_hiden_layer;
	int activationSize;
	int weightSize;
	float mutation_multiplayer;
};

#endif //!__NEURAL_NETWORK_DATA__
//Peer Programming: Guo, Albarello
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
	bool is_classification;
	double alpha;
};

struct NeuralNetworkDataCompact
{
	int self_l;
	bool is_classification;
	double alpha;
};

#endif //!__NEURAL_NETWORK_DATA__
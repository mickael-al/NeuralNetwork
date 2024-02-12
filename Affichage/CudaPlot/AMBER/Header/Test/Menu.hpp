//Peer Programming: Guo, Albarello
#ifndef __MENU__
#define __MENU__

#include "Scene.hpp"
#include "GameEngine.hpp"
#include "implot.h"
#include "../../../CudaNeuralNetwork/CudaNeuralNetwork.hpp"
#include <thread>
#include <string>

class Menu : public Scene, public ImguiBlock, public Behaviour
{
public:
	void load();
	void unload();
	void preRender(VulkanMisc* vM);
	void render(VulkanMisc* vM);
	void start();
	void fixedUpdate();
	void update();
	void stop();
	void onGUI();
	void trainingData(std::vector<std::vector<double>> * data,std::vector<std::vector<double>> * result_data);
	void trainingLinearData(std::vector<glm::vec2>* data, std::vector<double>* result_data);
	static void TrainNN(NeuralNetwork* m_nn, TrainingNeuralNetworkInput* trainingNeuralNetworkInput, std::vector<std::vector<double>> xor_data, std::vector<std::vector<double>> xor_result_data, std::vector<float>* error, double * min_percent_error_train,bool * hasFinished);
	static void TrainDataSetNN(NeuralNetwork* m_nn, TrainingNeuralNetwork* trainingNeuralNetwork, std::string path, std::vector<float>* error, double* min_percent_error_train, bool* hasFinished);
private:
	ptrClass m_pc;
	void * m_Dll;
	std::vector<float> m_error;
	std::thread * m_currentThread = nullptr;
	std::string m_filepath = "./model.model";	
	std::string m_datapath = "./data_64.dataset";
	std::string m_testpath = "./imageTest";
	bool m_trainingState = false;
	int selectedTestCase = 0;
	double m_min_percent_error_train = 0.05f;
	std::vector<std::vector<std::vector<float>>> m_class;
	std::vector<std::string*> m_name_class;
	NeuralNetworkData m_nnd;
	double m_gamma = 10;
	NeuralNetwork* m_nn = nullptr;
	LinearModel* m_lm = nullptr;
	Rbf* m_rbf = nullptr;
	CreateNeuralNetwork m_createNeuralNetwork;
	UpdateNNAlpha m_updateNNAlpha;
	ReleaseNeuralNetwork m_releaseNeuralNetwork;
	TrainingNeuralNetworkInput m_trainingNeuralNetworkInput;
	TrainingNeuralNetwork m_trainingNeuralNetwork;
	GenerateDataSet m_generateDataSet;
	UseNeuralNetworkInput m_useNeuralNetworkInput;
	LoadNeuralNetworkModel m_loadNeuralNetworkModel;
	SaveNeuralNetworkModel m_saveNeuralNetworkModel;
	UseNeuralNetworkImage m_useNeuralNetworkImage;

	CreateLinearModel m_linearModel;
	ReleaseLinearModel m_releaseLinearModel;
	TrainingLinearModel m_trainingLinearModel;
	PredictLinearModel m_predictLinearModel;

	CreateRbf m_createRbf;
	ReleaseRbf m_releaseRbf;
	PredictRbf m_predictRbf;	
};

#endif //!__MENU__
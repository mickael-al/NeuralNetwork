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
	void trainingData(std::vector<std::vector<float>> * data,std::vector<std::vector<float>> * result_data);
	void trainingLinearData(std::vector<glm::vec2>* data, std::vector<double>* result_data);
	static void TrainNN(NeuralNetwork* m_nn, TrainingNeuralNetworkInput* trainingNeuralNetworkInput, std::vector<std::vector<float>> xor_data, std::vector<std::vector<float>> xor_result_data, std::vector<float>* error, float * min_percent_error_train,bool * hasFinished);
private:
	ptrClass m_pc;
	void * m_Dll;
	std::vector<float> m_error;
	std::thread * m_currentThread = nullptr;
	std::string m_filepath = "./model.model";
	bool m_trainingState = false;
	int selectedTestCase = 0;
	float m_min_percent_error_train = 0.05f;
	std::vector<std::vector<std::vector<float>>> m_class;
	std::vector<std::string*> m_name_class;
	NeuralNetworkData m_nnd;
	NeuralNetwork* m_nn = nullptr;
	LinearModel* m_lm = nullptr;

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
};

#endif //!__MENU__
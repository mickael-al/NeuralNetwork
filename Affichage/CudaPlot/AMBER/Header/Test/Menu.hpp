#ifndef __MENU__
#define __MENU__

#include "Scene.hpp"
#include "GameEngine.hpp"
#include "implot.h"
#include "../../../CudaNeuralNetwork/CudaNeuralNetwork.hpp"

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
private:
	ptrClass m_pc;
	void * m_Dll;
	CreateNeuralNetwork m_createNeuralNetwork;
	ReleaseNeuralNetwork m_releaseNeuralNetwork;
	TrainingNeuralNetworkInput m_trainingNeuralNetworkInput;
	TrainingNeuralNetwork m_trainingNeuralNetwork;
	GenerateDataSet m_generateDataSet;
	UseNeuralNetworkInput m_useNeuralNetworkInput;
	LoadNeuralNetworkModel m_loadNeuralNetworkModel;
	SaveNeuralNetworkModel m_saveNeuralNetworkModel;
	UseNeuralNetworkImage m_useNeuralNetworkImage;
};

#endif //!__MENU__
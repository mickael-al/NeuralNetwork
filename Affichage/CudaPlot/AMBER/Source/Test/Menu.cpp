#include <Windows.h>
#include "Menu.hpp"
#include <iostream>
#include "../../../CudaNeuralNetwork/CudaNeuralNetwork.hpp"

void Menu::load()
{
	m_pc = GameEngine::getPtrClass();
	m_pc.hud->addBlockUI(this);

    HMODULE hDll = LoadLibrary("..\\..\\..\\CudaNeuralNetwork\\x64\\Release\\CudaNeuralNetwork.dll");
    if (hDll == NULL)
    {
        return;
    }

    CreateNeuralNetwork createNeuralNetwork = (CreateNeuralNetwork)GetProcAddress(hDll, "createNeuralNetwork");
    ReleaseNeuralNetwork releaseNeuralNetwork = (ReleaseNeuralNetwork)GetProcAddress(hDll, "releaseNeuralNetwork");
    TrainingNeuralNetworkInput trainingNeuralNetworkInput = (TrainingNeuralNetworkInput)GetProcAddress(hDll, "trainingNeuralNetworkInput");
    TrainingNeuralNetwork trainingNeuralNetwork = (TrainingNeuralNetwork)GetProcAddress(hDll, "trainingNeuralNetwork");
    GenerateDataSet generateDataSet = (GenerateDataSet)GetProcAddress(hDll, "generateDataSet");
    UseNeuralNetworkInput useNeuralNetworkInput = (UseNeuralNetworkInput)GetProcAddress(hDll, "useNeuralNetworkInput");
    LoadNeuralNetworkModel loadNeuralNetworkModel = (LoadNeuralNetworkModel)GetProcAddress(hDll, "loadNeuralNetworkModel");
    SaveNeuralNetworkModel saveNeuralNetworkModel = (SaveNeuralNetworkModel)GetProcAddress(hDll, "saveNeuralNetworkModel");
    UseNeuralNetworkImage useNeuralNetworkImage = (UseNeuralNetworkImage)GetProcAddress(hDll, "useNeuralNetworkImage");
    if (createNeuralNetwork == NULL)
    {
        std::cerr << "createNeuralNetwork not found" << std::endl;
        return;
    }
    if (releaseNeuralNetwork == NULL)
    {
        std::cerr << "releaseNeuralNetwork not found" << std::endl;
        return;
    }

    NeuralNetworkData nnd{};
    nnd.nb_input_layer = 2;
    nnd.nb_col_hiden_layer = 2;
    nnd.nb_hiden_layer = 512;
    nnd.nb_output_layer = 1;
    nnd.alpha = 0.005f;
    nnd.is_classification = false;
    NeuralNetwork* nn = createNeuralNetwork(nnd);
    std::vector<std::vector<float>> xor_data;
    xor_data.push_back({ 0,0 });
    xor_data.push_back({ 1,0 });
    xor_data.push_back({ 0,1 });
    xor_data.push_back({ 1,1 });
    std::vector<std::vector<float>> xor_result_data;
    xor_result_data.push_back({ -1 });
    xor_result_data.push_back({ 1 });
    xor_result_data.push_back({ 1 });
    xor_result_data.push_back({ -1 });
    trainingNeuralNetworkInput(nn, xor_data, xor_result_data, 0.01f);
    //saveNeuralNetworkModel(nn, xorModelPath);
    //loadNeuralNetworkModel(nn, xorModelPath);
    useNeuralNetworkInput(nn, xor_data, &xor_result_data);
    for (int i = 0; i < xor_result_data.size(); i++)
    {
        for (int j = 0; j < xor_result_data[i].size(); j++)
        {
            std::cout << "Result : " << xor_result_data[i][j];
        }
        std::cout << std::endl;
    }
    releaseNeuralNetwork(nn);
}

void Menu::unload()
{    
	m_pc.hud->removeBlockUI(this);
}

void Menu::start()
{

}

void Menu::fixedUpdate()
{

}

void Menu::update()
{

}

void Menu::stop()
{

}

void Menu::onGUI()
{

}

void Menu::preRender(VulkanMisc* vM)
{

}

void Menu::render(VulkanMisc* vM)
{
    if (ImPlot::BeginPlot("Xor Test", "X", "Y")) {
        // Create some sample data
        const double x[] = { 1.0, -1.0};
        const double y[] = { 1.0, -1.0};
        const double x1[] = {1.0, -1.0};
        const double y1[] = {-1.0, 1.0};

        // Plot the scatter plot
        ImPlot::PlotScatter("False", x, y, 2);
        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 6, ImPlot::GetColormapColor(1), IMPLOT_AUTO, ImPlot::GetColormapColor(1));
        ImPlot::PlotScatter("True", x1, y1, 2);
        ImPlot::PlotLine("Line", &x1[0], &y1[0], 2, 1, 0, sizeof(double));

        // End the plot
        ImPlot::EndPlot();
    }

}
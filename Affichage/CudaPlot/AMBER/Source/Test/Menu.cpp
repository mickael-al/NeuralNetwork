#include <Windows.h>
#include "Menu.hpp"
#include <iostream>

void Menu::load()
{
	m_pc = GameEngine::getPtrClass();
	m_pc.hud->addBlockUI(this);

    HMODULE m_hDll = LoadLibrary(".\\CudaNeuralNetwork.dll");
    if (m_hDll == NULL)
    {
        return;
    }
    m_Dll = (void*)m_hDll;

    m_createNeuralNetwork = (CreateNeuralNetwork)GetProcAddress(m_hDll, "createNeuralNetwork");
    m_releaseNeuralNetwork = (ReleaseNeuralNetwork)GetProcAddress(m_hDll, "releaseNeuralNetwork");
    m_trainingNeuralNetworkInput = (TrainingNeuralNetworkInput)GetProcAddress(m_hDll, "trainingNeuralNetworkInput");
    m_trainingNeuralNetwork = (TrainingNeuralNetwork)GetProcAddress(m_hDll, "trainingNeuralNetwork");
    m_generateDataSet = (GenerateDataSet)GetProcAddress(m_hDll, "generateDataSet");
    m_useNeuralNetworkInput = (UseNeuralNetworkInput)GetProcAddress(m_hDll, "useNeuralNetworkInput");
    m_loadNeuralNetworkModel = (LoadNeuralNetworkModel)GetProcAddress(m_hDll, "loadNeuralNetworkModel");
    m_saveNeuralNetworkModel = (SaveNeuralNetworkModel)GetProcAddress(m_hDll, "saveNeuralNetworkModel");
    m_useNeuralNetworkImage = (UseNeuralNetworkImage)GetProcAddress(m_hDll, "useNeuralNetworkImage");
    if (m_createNeuralNetwork == NULL)
    {
        std::cerr << "createNeuralNetwork not found" << std::endl;
        return;
    }
    if (m_releaseNeuralNetwork == NULL)
    {
        std::cerr << "releaseNeuralNetwork not found" << std::endl;
        return;
    }

    NeuralNetworkData nnd{};
    nnd.nb_input_layer = 2;
    nnd.nb_col_hiden_layer = 2;
    nnd.nb_hiden_layer = 2;
    nnd.nb_output_layer = 1;
    nnd.alpha = 0.1f;
    nnd.is_classification = false;
    NeuralNetwork* nn = m_createNeuralNetwork(nnd);
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
    std::vector<float> error;
    float min_percent_error_train = 0.05f;
    m_trainingNeuralNetworkInput(nn, xor_data, xor_result_data, &error, min_percent_error_train);
    for (int i = 0; i < error.size(); i++)
    {
        std::cout << error[i] << std::endl;
    }
    //m_saveNeuralNetworkModel(nn, xorModelPath);
    //m_loadNeuralNetworkModel(nn, xorModelPath);
    m_useNeuralNetworkInput(nn, xor_data, &xor_result_data);
    for (int i = 0; i < xor_result_data.size(); i++)
    {
        for (int j = 0; j < xor_result_data[i].size(); j++)
        {
            std::cout << "Result : " << xor_result_data[i][j];
        }
        std::cout << std::endl;
    }
    m_releaseNeuralNetwork(nn);
}

void Menu::unload()
{    
	m_pc.hud->removeBlockUI(this);
    HMODULE m_hDll = (HMODULE)m_Dll;
    if (m_hDll != nullptr) 
    {
        if (FreeLibrary(m_hDll)) 
        {
            std::cout << "DLL libérée avec succès." << std::endl;
        }
        else 
        {
            std::cerr << "Erreur lors de la libération de la DLL." << std::endl;
        }
    }
    else 
    {
        std::cerr << "Erreur lors du chargement de la DLL." << std::endl;
    }
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
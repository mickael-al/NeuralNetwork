//Peer Programming: Guo, Albarello
#include <iostream>
#include <Windows.h>
#include "CudaNeuralNetwork.hpp"

int main()
{
    HMODULE hDll = LoadLibrary(L".\\CudaNeuralNetwork.dll");
    if (hDll == NULL)
    {
        return 1;
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
    CreateLinearModel m_linearModel = (CreateLinearModel)GetProcAddress(hDll, "createLinearModel");
    ReleaseLinearModel  m_releaseLinearModel = (ReleaseLinearModel)GetProcAddress(hDll, "releaseLinearModel");
    TrainingLinearModel m_trainingLinearModel = (TrainingLinearModel)GetProcAddress(hDll, "trainingLinearModel");
    PredictLinearModel m_predictLinearModel = (PredictLinearModel)GetProcAddress(hDll, "predictLinearModel");
    if (createNeuralNetwork == NULL)
    {
        std::cerr << "createNeuralNetwork not found" << std::endl;
        return 1;
    }
    if (releaseNeuralNetwork == NULL)
    {
        std::cerr << "releaseNeuralNetwork not found" << std::endl;
        return 1;
    }
    const int image_size = 64;
    const std::string dataSetPath = "../DataSet/data_64_col.dataset";
    const std::string xorModelPath = "./xor_model.model";
    const std::string imageModelPath = "./image_model.model";
    const std::string imageTest = "./cat.png";
    generateDataSet("../ds2", dataSetPath, image_size);
    return 0;


    /*std::vector<Vec2> m_input;
    std::vector<double> m_result;
    m_input.push_back({ 1, 1 });
    m_input.push_back({ 2, 3 });
    m_input.push_back({ 3, 3 });
    m_result.push_back(1);
    m_result.push_back(-1);
    m_result.push_back(-1);
    std::vector<double> m_error;
    LinearModel * lm = m_linearModel();
    m_trainingLinearModel(lm, 0.01f, (Vec2*)(&m_input[0]), m_input.size(), m_result, &m_error);

    m_releaseLinearModel(lm);

    return 0;*/
    if (false)
    {
        NeuralNetworkData nnd{};
        nnd.nb_input_layer = image_size * image_size * 3;
        nnd.nb_col_hiden_layer = 2;
        nnd.nb_hiden_layer = 512;
        nnd.nb_output_layer = 3;
        nnd.alpha = 0.001f;
        nnd.is_classification = false;
        NeuralNetwork* nn = createNeuralNetwork(nnd);
        //trainingNeuralNetwork(nn, dataSetPath, 3.0);
        //saveNeuralNetworkModel(nn, imageModelPath);
        loadNeuralNetworkModel(nn, imageModelPath);
        std::vector<double> output;
        useNeuralNetworkImage(nn, "./13.jpg", &output);
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << "Result : " << output[j] << std::endl;
        }
        std::cout << std::endl;
        useNeuralNetworkImage(nn, "./14.jpg", &output);
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << "Result : " << output[j] << std::endl;
        }
        useNeuralNetworkImage(nn, "./15.jpg", &output);
        std::cout << std::endl;
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << "Result : " << output[j] << std::endl;
        }
        useNeuralNetworkImage(nn, "./139.jpg", &output);
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << "Result : " << output[j] << std::endl;
        }
        std::cout << std::endl;
        useNeuralNetworkImage(nn, "./142.jpg", &output);
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << "Result : " << output[j] << std::endl;
        }
        useNeuralNetworkImage(nn, "./144.jpg", &output);
        std::cout << std::endl;
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << "Result : " << output[j] << std::endl;
        }
        releaseNeuralNetwork(nn);
    }
    else
    {
        NeuralNetworkData nnd{};
        nnd.nb_input_layer = 2;
        nnd.nb_col_hiden_layer = 2;
        nnd.nb_hiden_layer = 2;
        nnd.nb_output_layer = 1;
        nnd.alpha = 0.1f;
        nnd.is_classification = false;
        NeuralNetwork* nn = createNeuralNetwork(nnd);
        std::vector<std::vector<double>> xor_data;
        xor_data.push_back({ 0,0 });
        xor_data.push_back({ 1,0 });
        xor_data.push_back({ 0,1 });
        xor_data.push_back({ 1,1 });
        std::vector<std::vector<double>> xor_result_data;
        xor_result_data.push_back({ -1 });
        xor_result_data.push_back({ 1 });
        xor_result_data.push_back({ 1 });
        xor_result_data.push_back({ -1 });
        std::vector<float> error;
        double min_percent = 0.01f;
        trainingNeuralNetworkInput(nn, xor_data, xor_result_data, &error, &min_percent);
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
    return 0;
}
//Peer Programming: Guo, Albarello

#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;
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
    m_updateNNAlpha = (UpdateNNAlpha)GetProcAddress(m_hDll,"updateNNAlpha");

    m_linearModel = (CreateLinearModel)GetProcAddress(m_hDll, "createLinearModel");
    m_releaseLinearModel = (ReleaseLinearModel)GetProcAddress(m_hDll, "releaseLinearModel");
    m_trainingLinearModel = (TrainingLinearModel)GetProcAddress(m_hDll, "trainingLinearModel");
    m_predictLinearModel = (PredictLinearModel)GetProcAddress(m_hDll, "predictLinearModel");
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
    m_nnd.nb_input_layer = 2;
    m_nnd.nb_col_hiden_layer = 2;
    m_nnd.nb_hiden_layer = 2;
    m_nnd.nb_output_layer = 1;
    m_nnd.alpha = 0.1f;
    m_nnd.is_classification = false;
    std::srand(static_cast<unsigned int>(std::time(0)));
    m_filepath.resize(256);
    m_datapath.resize(256);
    m_testpath.resize(256);
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

void Menu::TrainNN(NeuralNetwork* m_nn, TrainingNeuralNetworkInput * trainingNeuralNetworkInput, std::vector<std::vector<float>> xor_data, std::vector<std::vector<float>> xor_result_data, std::vector<float> * error,float * min_percent_error_train, bool * hasFinished)
{    
    *hasFinished = true;
    (*trainingNeuralNetworkInput)(m_nn, xor_data, xor_result_data, error, min_percent_error_train);
    *hasFinished = false;
}

void Menu::TrainDataSetNN(NeuralNetwork* m_nn, TrainingNeuralNetwork* trainingNeuralNetwork, std::string path, std::vector<float>* error, float* min_percent_error_train, bool* hasFinished)
{
    *hasFinished = true;
    (*trainingNeuralNetwork)(m_nn, path, error, min_percent_error_train);
    *hasFinished = false;
}

void getArborescence(const fs::path& chemin, const fs::path& basePath, std::map<std::string, std::vector<std::string>> * data)
{
    if (fs::is_directory(chemin))
    {
        for (const auto& entry : fs::directory_iterator(chemin))
        {
            if (fs::is_regular_file(entry.path()))
            {
                std::string extension = entry.path().extension().string();
                if (extension == ".jpg" || extension == ".png")
                {
                    std::string s = chemin.string();
                    s = s.substr(basePath.string().length() + 1, s.length());
                    size_t pos = s.find("\\");
                    s = s.substr(0, pos);
                    (*data)[s].push_back(entry.path().string().c_str());
                }
            }
            if (fs::is_directory(entry.path()))
            {
                getArborescence(entry.path(), basePath, data);
            }
        }
    }
}

const char* CasTest[] = { "LinearSimple","LinearMultiple","Xor","Cross","MultiLinear","MultiCross"};

void Menu::trainingLinearData(std::vector<glm::vec2>* data, std::vector<double>* result_data)
{
    if (selectedTestCase == 0)
    {
        (*data).push_back({ 1,1 });
        (*data).push_back({ 2,3 });
        (*data).push_back({ 3,3 });
        (*result_data).push_back({ 1 });
        (*result_data).push_back({ -1 });
        (*result_data).push_back({ -1 });
    }
    else if (selectedTestCase == 1)
    {
        for (int i = 0; i < 50; i++)
        {
            float x1 = (std::rand() % 1000 / 1000.0) * 0.9 + 1;
            float y1 = (std::rand() % 1000 / 1000.0) * 0.9 + 1;
            (*data).push_back({ x1,y1 });
            (*result_data).push_back({ 1.0f });
            float x2 = (std::rand() % 1000 / 1000.0) * 0.9 + 2;
            float y2 = (std::rand() % 1000 / 1000.0) * 0.9 + 2;
            (*data).push_back({ x2,y2 });
            (*result_data).push_back({ -1.0f });
        }
    }
    else if (selectedTestCase == 2)
    {
        (*data).push_back({ 0,0 });
        (*data).push_back({ 1,0 });
        (*data).push_back({ 0,1 });
        (*data).push_back({ 1,1 });
        (*result_data).push_back({ -1 });
        (*result_data).push_back({ 1 });
        (*result_data).push_back({ 1 });
        (*result_data).push_back({ -1 });
    }
    else if (selectedTestCase == 3)
    {
        for (int i = 0; i < 500; ++i)
        {
            float x = (std::rand() % 2000 / 1000.0) - 1.0;
            float y = (std::rand() % 2000 / 1000.0) - 1.0;
            (*data).push_back({ x, y });

            float label = (std::abs(x) <= 0.3 || std::abs(y) <= 0.3) ? 1 : -1;
            (*result_data).push_back({ label });
        }
    }    
}

void Menu::trainingData(std::vector<std::vector<float>>* data, std::vector<std::vector<float>>* result_data)
{
    if (selectedTestCase == 0)
    {
        (*data).push_back({ 1,1 });
        (*data).push_back({ 2,3 });
        (*data).push_back({ 3,3 });
        (*result_data).push_back({ 1 });
        (*result_data).push_back({ -1 });
        (*result_data).push_back({ -1 });
    }
    else if (selectedTestCase == 1)
    {
        for (int i = 0; i < 50; i++)
        {
            float x1 = (std::rand() % 1000 / 1000.0) * 0.9 + 1;
            float y1 = (std::rand() % 1000 / 1000.0) * 0.9 + 1;
            (*data).push_back({ x1,y1 });
            (*result_data).push_back({ 1.0f });
            float x2 = (std::rand() % 1000 / 1000.0) * 0.9 + 2;
            float y2 = (std::rand() % 1000 / 1000.0) * 0.9 + 2;
            (*data).push_back({ x2,y2 });
            (*result_data).push_back({ -1.0f });
        }
    }
    else if (selectedTestCase == 2)
    {
        (*data).push_back({ 0,0 });
        (*data).push_back({ 1,0 });
        (*data).push_back({ 0,1 });
        (*data).push_back({ 1,1 });
        (*result_data).push_back({ -1 });
        (*result_data).push_back({ 1 });
        (*result_data).push_back({ 1 });
        (*result_data).push_back({ -1 });
    }
    else if (selectedTestCase == 3)
    {
        for (int i = 0; i < 500; ++i)
        {
            float x = (std::rand() % 2000 / 1000.0) - 1.0;
            float y = (std::rand() % 2000 / 1000.0) - 1.0;
            (*data).push_back({ x, y });

            float label = (std::abs(x) <= 0.3 || std::abs(y) <= 0.3) ? 1 : -1;
            (*result_data).push_back({ label });
        }
    }
    else if (selectedTestCase == 4)
    {
        for (int i = 0; i < 500; ++i)
        {
            float x = (std::rand() % 2000 / 1000.0) - 1.0;
            float y = (std::rand() % 2000 / 1000.0) - 1.0;
            (*data).push_back({ x, y });
        }

        for (const auto& p : (*data))
        {
            std::vector<float> label;
            if (-p[0] - p[1] - 0.5 > 0 && p[1] < 0 && p[0] - p[1] - 0.5 < 0)
            {
                label = { 1, 0, 0 };
            }
            else if (-p[0] - p[1] - 0.5 < 0 && p[1] > 0 && p[0] - p[1] - 0.5 < 0)
            {
                label = { 0, 1, 0 };
            }
            else if (-p[0] - p[1] - 0.5 < 0 && p[1] < 0 && p[0] - p[1] - 0.5 > 0)
            {
                label = { 0, 0, 1 };
            }
            else
            {
                label = { 0, 0, 0 };
            }
            (*result_data).push_back(label);
        }

        std::vector<std::vector<float>> filteredX;
        std::vector<std::vector<float>> filteredY;

        for (size_t i = 0; i < (*result_data).size(); ++i)
        {
            if ((*result_data)[i][0] != 0 || (*result_data)[i][1] != 0 || (*result_data)[i][2] != 0)
            {
                filteredX.push_back((*data)[i]);
                filteredY.push_back((*result_data)[i]);
            }
        }
        (*data) = filteredX;
        (*result_data) = filteredY;
    }
    else if (selectedTestCase == 5)
    {
        for (int i = 0; i < 1000; ++i)
        {
            float x = (std::rand() % 2000 / 1000.0) - 1.0;
            float y = (std::rand() % 2000 / 1000.0) - 1.0;
            (*data).push_back({ x, y });
        }

        for (const auto& p : (*data))
        {
            float x_mod = std::fmod(std::abs(p[0]), 0.5);
            float y_mod = std::fmod(std::abs(p[1]), 0.5);
            std::vector<float> label;

            if (x_mod <= 0.25 && y_mod > 0.25)
            {
                label = { 1, 0, 0 };
            }
            else if (x_mod > 0.25 && y_mod <= 0.25)
            {
                label = { 0, 1, 0 };
            }
            else
            {
                label = { 0, 0, 1 };
            }

            (*result_data).push_back(label);
        }
    }

}

void Menu::render(VulkanMisc* vM)
{
    ImGui::SetNextWindowSize(ImVec2((float)m_pc.settingManager->getWindowWidth(), (float)m_pc.settingManager->getWindowHeight()));
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::Begin("Dll Test");
    ImGui::Text("NN Setting");
    ImGui::DragInt("Input layer Size", &(m_nnd.nb_input_layer));
    ImGui::DragInt("Hiden layer Size", &m_nnd.nb_hiden_layer);
    ImGui::DragInt("Hiden Col layer Size", &m_nnd.nb_col_hiden_layer);
    ImGui::DragInt("Output layer Size", &m_nnd.nb_output_layer);
    if (ImGui::DragFloat("alpha", &m_nnd.alpha) && m_nn != nullptr)
    {
        m_updateNNAlpha(m_nn, m_nnd.alpha);
    }
    ImGui::Checkbox("is classification", &m_nnd.is_classification);
    ImGui::Text("Training Setting");
    ImGui::DragFloat("minimum percent error train", &m_min_percent_error_train);
    ImGui::InputText("Neural Network path", &m_filepath[0], 256);
    ImGui::InputText("Neural DataSet path", &m_datapath[0], 256);
    ImGui::InputText("Neural Image path", &m_testpath[0], 256);
    if (ImGui::BeginCombo("Test Case", CasTest[selectedTestCase]))
    {
        for (int i = 0; i < IM_ARRAYSIZE(CasTest); i++)
        {
            bool isSelected = (selectedTestCase == i);
            if (ImGui::Selectable(CasTest[i], isSelected))
            {
                selectedTestCase = i;  
            }

            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();                
            }
        }
        ImGui::EndCombo();
    }


    if (m_lm != nullptr)
    {
        if (ImGui::Button("Delete linear"))
        {
            m_releaseLinearModel(m_lm);
            m_lm = nullptr;
        }
        ImGui::SameLine();
        if (ImGui::Button("Training linear Input"))
        {
            std::vector<glm::vec2> data;
            std::vector<double> result_data;
            trainingLinearData(&data, &result_data);
            m_error.clear();
            m_trainingLinearModel(m_lm, m_nnd.alpha, (Vec2*)(&data[0]), data.size(), result_data,&m_error);
        }
        ImGui::SameLine();
        if (ImGui::Button("Result"))
        {
            std::vector<glm::vec2> data;
            std::vector<double> result_data;
            trainingLinearData(&data, &result_data);
            m_class.clear();
            m_class.resize(2);
            for (int i = 0; i < m_name_class.size(); i++)
            {
                delete m_name_class[i];
            }
            m_name_class.clear();
            for (int k = 0; k < m_class.size(); k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    m_class[k].push_back({});
                }
                m_name_class.push_back(new std::string("Class" + std::to_string(k)));
            }
            for (int i = 0; i < data.size(); i++)
            {                
                int Class = result_data[i] >= 0 ? 1 : 0;
                for (int j = 0; j < 2; j++)
                {
                    m_class[Class][j].push_back(data[i][j]);
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Use"))
        {
            std::vector<glm::vec2> data;
            std::vector<double> result_data;
            trainingLinearData(&data, &result_data);            
            m_class.clear();
            m_class.resize(2);
            for (int i = 0; i < m_name_class.size(); i++)
            {
                delete m_name_class[i];
            }
            m_name_class.clear();
            for (int k = 0; k < m_class.size(); k++)
            {
                for (int j = 0; j < 2; j++)
                {
                    m_class[k].push_back({});
                }
                m_name_class.push_back(new std::string("Class" + std::to_string(k)));
            }
            for (int i = 0; i < data.size(); i++)
            {
                int Class = m_predictLinearModel(m_lm, (Vec2*)(&data[i])) >= 0 ? 1 : 0;
                for (int j = 0; j < 2; j++)
                {
                    m_class[Class][j].push_back(data[i][j]);
                }
            }
        }
    }
    else if (m_nn == nullptr)
    {
        if (ImGui::Button("Create NN"))
        {
            m_nn = m_createNeuralNetwork(m_nnd);
        }   
        ImGui::SameLine();
        if (ImGui::Button("Create linear"))
        {
            m_lm = m_linearModel();
        }
        ImGui::SameLine();
        if (ImGui::Button("Setup Image"))
        {
            m_nnd.nb_input_layer = 64 * 64 * 3;
            m_nnd.nb_hiden_layer = 512;
            m_nnd.nb_col_hiden_layer = 2;
            m_nnd.nb_output_layer = 3;
            m_nnd.alpha = 0.001;
        }
        ImGui::SameLine();
        if (ImGui::Button("Setup Input"))
        {
            m_nnd.nb_input_layer = 2;
            m_nnd.nb_output_layer = 1;
            m_nnd.alpha = 0.01;
        }
    }
    else
    {
        if (ImGui::Button("Delete NN") && m_nn != nullptr && !m_trainingState)
        {
            m_releaseNeuralNetwork(m_nn);
            m_nn = nullptr;
        }
        ImGui::SameLine();
        if (!m_trainingState)
        {
            if (ImGui::Button("Training DataSet"))
            {
                m_error.clear();
                m_currentThread = new std::thread(&Menu::TrainDataSetNN, m_nn, &m_trainingNeuralNetwork, m_datapath, &m_error, &m_min_percent_error_train, &m_trainingState);
                m_currentThread->detach();
            }
            ImGui::SameLine();
            if (ImGui::Button("Training Input"))
            {
                std::vector<std::vector<float>> data;
                std::vector<std::vector<float>> result_data;
                trainingData(&data, &result_data);
                m_error.clear();
                m_currentThread = new std::thread(&Menu::TrainNN, m_nn, &m_trainingNeuralNetworkInput, data, result_data, &m_error, &m_min_percent_error_train,&m_trainingState);
                m_currentThread->detach();
            }
            ImGui::SameLine();
            if (ImGui::Button("Result"))
            {
                std::vector<std::vector<float>> data;
                std::vector<std::vector<float>> result_data;
                trainingData(&data, &result_data);        
                m_class.clear();
                m_class.resize(result_data[0].size() + 1);
                for (int i = 0; i < m_name_class.size(); i++)
                {
                    delete m_name_class[i];
                }
                m_name_class.clear();
                for (int k = 0; k < m_class.size(); k++)
                {
                    for (int j = 0; j < data[0].size(); j++)
                    {
                        m_class[k].push_back({});
                    }
                    m_name_class.push_back(new std::string("Class" + std::to_string(k)));
                }
                for (int i = 0; i < data.size(); i++)
                {
                    int inter = 0;
                    for (int k = 1; k < result_data[i].size(); k++)
                    {
                        if (result_data[i][inter] < result_data[i][k])
                        {
                            inter = k;
                        }
                    }
                    int Class = result_data[i][inter] > 0 ? inter : result_data[i].size();                    
                    for (int j = 0; j < data[i].size(); j++)
                    {
                        m_class[Class][j].push_back(data[i][j]);
                    }
                }                               
            }            
            ImGui::SameLine();
            if (ImGui::Button("Use"))
            {
                std::vector<std::vector<float>> data;
                std::vector<std::vector<float>> result_data;
                trainingData(&data, &result_data);
                m_useNeuralNetworkInput(m_nn, data, &result_data);
                m_class.clear();
                m_class.resize(result_data[0].size() + 1);
                for (int i = 0; i < m_name_class.size(); i++)
                {
                    delete m_name_class[i];
                }
                m_name_class.clear();
                for (int k = 0; k < m_class.size(); k++)
                {
                    for (int j = 0; j < data[0].size(); j++)
                    {
                        m_class[k].push_back({});
                    }
                    m_name_class.push_back(new std::string("Class" + std::to_string(k)));
                }
                for (int i = 0; i < data.size(); i++)
                {
                    int inter = 0;
                    for (int k = 1; k < result_data[i].size(); k++)
                    {
                        if (result_data[i][inter] < result_data[i][k])
                        {
                            inter = k;
                        }
                    }
                    int Class = result_data[i][inter] > 0 ? inter : result_data[i].size();
                    for (int j = 0; j < data[i].size(); j++)
                    {
                        m_class[Class][j].push_back(data[i][j]);
                    }
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Result Image"))
            {
                std::map<std::string, std::vector<std::string>> data;                
                getArborescence("./ImageTest", "./ImageTest", &data);
                m_class.clear();
                m_class.resize(data.size());
                for (int i = 0; i < m_name_class.size(); i++)
                {
                    delete m_name_class[i];
                }
                m_name_class.clear();
                int k = 0;
                for (auto a : data)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        m_class[k].push_back({});
                    }
                    m_name_class.push_back(new std::string(a.first));
                    k++;
                }     
                k = 0;
                float last = 0.0f;
                for (auto a : data)
                {                       
                    for (int i = 0; i < a.second.size(); i++)
                    {                        
                        m_class[k][0].push_back((k* (last /30.0f)) + (i / 40));
                        m_class[k][1].push_back(i%40);
                    }
                    last = a.second.size();
                    k++;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Use Image"))
            {                
                std::map<std::string, std::vector<std::string>> data;
                getArborescence("./ImageTest", "./ImageTest", &data);
                m_class.clear();
                m_class.resize(data.size());
                for (int i = 0; i < m_name_class.size(); i++)
                {
                    delete m_name_class[i];
                }
                m_name_class.clear();
                int k = 0;
                for (auto a : data)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        m_class[k].push_back({});
                    }
                    m_name_class.push_back(new std::string(a.first));
                    k++;
                }
                k = 0;
                int nk = 0;
                float last = 0.0f;
                for (auto a : data)
                {
                    for (int i = 0; i < a.second.size(); i++)
                    {
                        std::vector<float> output;
                        m_useNeuralNetworkImage(m_nn, a.second[i], &output);
                        nk = 0;
                        for (int j = 1; j < data.size(); j++)
                        {
                            if (output[nk] < output[j])
                            {
                                nk = j;
                            }
                        }
                        m_class[nk][0].push_back((k * (last / 30.0f)) + (i / 40));
                        m_class[nk][1].push_back(i % 40);
                    }
                    last = a.second.size();
                    k++;
                }
            }
            if (ImGui::Button("Save NN"))
            {
                m_saveNeuralNetworkModel(m_nn, m_filepath);
            }
            ImGui::SameLine();
            if (ImGui::Button("Load NN"))
            {
                m_loadNeuralNetworkModel(m_nn, m_filepath);
            }
        }
        else
        {
            if (ImGui::Button("Stop Training"))
            {
                float vi = m_min_percent_error_train;
                m_min_percent_error_train = 100.0f;
                while (m_trainingState) { Debug::Log("Wait"); }
                m_min_percent_error_train = vi;
                m_currentThread = nullptr;
            }
        }
    }
    ImPlot::SetNextAxesLimits(0, m_error.size(), 0, 100.0, ImGuiCond_Always);
    if (ImPlot::BeginPlot("Learning curve", "Step", "Error percent")) 
    {
        if (m_error.size() > 0)
        {
            ImPlot::PlotLine("Learning curve", &m_error[0], m_error.size());            
        }
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("Result","X","Y"))
    {
        if (m_class.size() > 0)
        {           
            for (int i = 0; i < m_class.size(); i++)
            {
                ImPlot::SetNextMarkerStyle(i%10, 6, ImPlot::GetColormapColor(i), IMPLOT_AUTO, ImPlot::GetColormapColor(i));
                ImPlot::PlotScatter(m_name_class[i]->c_str(), m_class[i][0].data(), m_class[i][1].data(), m_class[i][0].size());
            }
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}
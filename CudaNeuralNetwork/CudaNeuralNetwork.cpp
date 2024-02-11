//Peer Programming: Guo, Albarello
#include "CudaNeuralNetwork.hpp"
#include "CNNHelper.hpp"
#include "NeuralNetwork.hpp"
#include "kernel.h"
#include <stdio.h>
#include "ImageData.hpp"
#include <map>
#include <filesystem>
#include <fstream>
#include "LinearModel.hpp"

namespace fs = std::filesystem;

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd)
{
    if (nnd.nb_input_layer <= 0 || nnd.nb_col_hiden_layer <= 0 || nnd.nb_hiden_layer <= 0 || nnd.nb_output_layer <= 0)
    {
        fprintf(stderr, "createNeuralNetwork failed! invalid input NeuralNetworkData\n");
        return nullptr;
    }
    std::cout << "Create Neural network" << std::endl;
    return new NeuralNetwork(nnd);
}

std::map<const std::string, std::vector<double*>> charger(const std::string& nom_fichier, int* size)
{
    std::map<const std::string, std::vector<double*>> donnees;
    std::ifstream fichier(nom_fichier, std::ios::binary);
    if (fichier.is_open())
    {
        // Lecture de la taille de la map
        size_t taille_map;
        size_t image_size;
        fichier.read(reinterpret_cast<char*>(&taille_map), sizeof(size_t));
        fichier.read(reinterpret_cast<char*>(&image_size), sizeof(size_t));
        *size = image_size;
        // Lecture des donn�es pour chaque paire cl�-valeur dans la map
        for (size_t i = 0; i < taille_map; ++i) {
            // Lecture de la taille de la cl�
            size_t taille_cle;
            fichier.read(reinterpret_cast<char*>(&taille_cle), sizeof(size_t));
            // Lecture de la cl�
            char* cle = new char[taille_cle + 1];
            fichier.read(cle, taille_cle);
            cle[taille_cle] = '\0';

            // Lecture de la taille du vecteur
            size_t taille_vecteur;
            fichier.read(reinterpret_cast<char*>(&taille_vecteur), sizeof(size_t));

            // Lecture des donn�es double du vecteur
            std::vector<double*> vecteur;
            for (size_t j = 0; j < taille_vecteur; ++j) {
                double* valeur = new double[image_size * image_size * 3];
                fichier.read(reinterpret_cast<char*>(valeur), sizeof(double) * image_size * image_size * 3);
                vecteur.push_back(valeur);
            }

            // Stockage des donn�es dans la map
            donnees[std::string(cle)] = vecteur;
            delete[] cle;
        }
        fichier.close();
        std::cout << "Donn�es charg�es avec succ�s depuis " << nom_fichier << std::endl;
    }
    else {
        std::cerr << "Impossible d'ouvrir le fichier pour chargement." << std::endl;
    }
    return donnees;
}

LinearModel* createLinearModel()
{
    return new LinearModel();
}

void releaseLinearModel(LinearModel* lm)
{
    delete lm;
}

void trainingLinearModel(LinearModel* lm,double learning_rate, Vec2* training_data, int size, std::vector<double> point, std::vector<float>* error)
{
    lm->training(learning_rate,training_data,size,point,error);
}

double predictLinearModel(LinearModel* lm,Vec2* point)
{
    return lm->predict(point);
}

void trainingNeuralNetwork(NeuralNetwork* nn, const std::string& dataSetPath, std::vector<float>* error, double * min_percent_error_train)
{
    int size;
    std::map<const std::string, std::vector<double*>> data = charger(dataSetPath,&size);
    nn->trainingDataSet(data, size, error, min_percent_error_train);    
}

void trainingNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<double>> input, const std::vector<std::vector<double>> output, std::vector<float>* error, double * min_percent_error_train)
{
    nn->trainingInput(input, output,error, min_percent_error_train);
}

void updateNNAlpha(NeuralNetwork* nn, double alpha)
{
    nn->updateAlpha(alpha);
}

void loadNeuralNetworkModel(NeuralNetwork* nn, const std::string& modelPath) 
{
    nn->loadModel(modelPath);
}

void saveNeuralNetworkModel(NeuralNetwork* nn, const std::string& modelPath)
{
    nn->saveModel(modelPath);
}

void useNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<double>> input, std::vector<std::vector<double>> * ouput)
{
    nn->useInput(input, ouput);
}

void useNeuralNetworkImage(NeuralNetwork* nn, const std::string& image_path, std::vector<double>* output)
{
    NeuralNetworkData* nnd = nn->getNeuralNetworkData();
    ImageData* id = new ImageData(image_path.c_str());
    int csize = (int)sqrt(nnd->nb_input_layer);
    if (id->getHeight() != csize || id->getWidth() != csize)
    {        
        id->resize(csize);
        id->write();
    }    
    stbimg stb = ImageData::loadData(id->getPath());
    delete id;
    double* col = new double[csize * csize * 3];

    int offset = 0;
    int offset2 = 0;
    if (stb.ch == 3)
    {
        for (int i = 0; i < stb.height; i++)
        {
            for (int j = 0; j < stb.width; j++)
            {
                offset = (i * stb.width + j) * stb.ch;
                col[offset] = (((double)stb.data[offset] / 255.0) * 2.0) - 1.0;
                col[offset + 1] = (((double)stb.data[offset + 1] / 255.0) * 2.0) - 1.0;
                col[offset + 2] = (((double)stb.data[offset + 2] / 255.0) * 2.0) - 1.0;
            }
        }
    }
    else if (stb.ch == 4)
    {
        for (int i = 0; i < stb.height; i++)
        {
            for (int j = 0; j < stb.width; j++)
            {
                offset = (i * stb.width + j) * 4;
                offset2 = (i * stb.width + j) * 3;
                col[offset2] = (((double)stb.data[offset] / 255.0) * 2.0) - 1.0;
                col[offset2 + 1] = (((double)stb.data[offset + 1] / 255.0) * 2.0) - 1.0;
                col[offset2 + 2] = (((double)stb.data[offset + 2] / 255.0) * 2.0) - 1.0;
            }
        }
    }

    nn->useInputImage(col, output);
}

void releaseNeuralNetwork(NeuralNetwork* network)
{
    delete network;
}

void getArborescence(const fs::path& chemin, const fs::path& basePath, std::map<const std::string, std::vector<stbimg>> * data,int size)
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
                    ImageData* id = new ImageData(entry.path().string().c_str());
                    if (id->getHeight() < size && id->getWidth() < size)
                    {
                        fs::remove(entry.path());
                        std::cout << " remove -> "  << entry.path() << std::endl;
                    }
                    else if(id->getHeight() != size || id->getWidth() != size)
                    {
                        std::cout << " modify -> " << entry.path() << std::endl;
                        id->resize(size);
                        id->write();                        
                    }
                    else
                    {
                        std::string s = chemin.string();
                        s=s.substr(basePath.string().length()+1,s.length());
                        size_t pos = s.find("\\");
                        s = s.substr(0, pos);
                        (*data)[s].push_back(ImageData::loadData(id->getPath()));
                    }
                    delete id;
                }                               
            }
            if (fs::is_directory(entry.path())) 
            {
                getArborescence(entry.path(), basePath, data, size);
            }
        }
    }
}

void sauvegarder(const std::map<const std::string, std::vector<double*>>& data, const std::string& nom_fichier, size_t image_size)
{
    std::ofstream fichier(nom_fichier, std::ios::binary);
    if (fichier.is_open()) 
    {
        // �criture de la taille de la map
        size_t taille_map = data.size();
        fichier.write(reinterpret_cast<const char*>(&taille_map), sizeof(size_t));
        // Taille d'une image 
        fichier.write(reinterpret_cast<const char*>(&image_size), sizeof(size_t));
        // Pour chaque paire cl�-valeur dans la map
        for (const auto& paire : data) 
        {
            // �criture de la taille de la cl�
            size_t taille_cle = paire.first.size();
            fichier.write(reinterpret_cast<const char*>(&taille_cle), sizeof(size_t));
            // �criture de la cl�
            fichier.write(paire.first.c_str(), taille_cle);

            // �criture de la taille du vecteur
            size_t taille_vecteur = paire.second.size();
            fichier.write(reinterpret_cast<const char*>(&taille_vecteur), sizeof(size_t));

            // �criture des donn�es double du vecteur
            for (size_t i = 0; i < taille_vecteur; ++i) 
            {
                fichier.write(reinterpret_cast<const char*>(paire.second[i]), sizeof(double)* image_size * image_size *3);
            }
        }
        fichier.close();
        std::cout << "Donn�es sauvegard�es avec succ�s dans " << nom_fichier << std::endl;
    }
    else {
        std::cerr << "Impossible d'ouvrir le fichier pour sauvegarde." << std::endl;
    }
}

void generateDataSet(const std::string& path, const std::string& dataSetSavepath,int image_data_size)
{
    std::map<const std::string, std::vector<stbimg>> m_map_dataset;
    std::cout << "load Image" << std::endl;
    getArborescence(path, path, &m_map_dataset, image_data_size);
    std::cout << "Compute ThanH Image" << std::endl;
    std::map<const std::string, std::vector<double*>> data;
    for (const auto& pair : m_map_dataset) 
    {        
        std::cout << pair.first << std::endl;
        std::vector<double*> d;
        for (const auto& value : pair.second) 
        {
            double* col = new double[image_data_size * image_data_size * 3];

            int offset = 0;
            int offset2 = 0;
            if (value.ch == 3)
            {
                for (int i = 0; i < value.height; i++)
                {
                    for (int j = 0; j < value.width; j++)
                    {
                        offset = (i * value.width + j) * value.ch;
                        col[offset] = (((double)value.data[offset] / 255.0) * 2.0) - 1.0;
                        col[offset + 1] = (((double)value.data[offset + 1] / 255.0) * 2.0) - 1.0;
                        col[offset + 2] = (((double)value.data[offset + 2] / 255.0) * 2.0) - 1.0;
                    }
                }
            }
            else if (value.ch == 4)
            {
                for (int i = 0; i < value.height; i++)
                {
                    for (int j = 0; j < value.width; j++)
                    {
                        offset = (i * value.width + j) * 4;
                        offset2 = (i * value.width + j) * 3;
                        col[offset2] = (((double)value.data[offset] / 255.0) * 2.0) - 1.0;
                        col[offset2 + 1] = (((double)value.data[offset + 1] / 255.0) * 2.0) - 1.0;
                        col[offset2 + 2] = (((double)value.data[offset + 2] / 255.0) * 2.0) - 1.0;
                    }
                }
            }
            d.push_back(col);
        }
        data[pair.first] = d;
        std::cout << std::endl;
    }
    std::cout << "Save" << std::endl;
    sauvegarder(data, dataSetSavepath, image_data_size);
    std::cout << "Finish" << std::endl;
}

int addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int* thread_size = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&thread_size, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(thread_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 dimGrid;
    dim3 dimBlock;

    CNNHelper::KernelDispath(size, &deviceProp, &dimGrid, &dimBlock);
    AddKernel(dimGrid, dimBlock,dev_c, dev_a, dev_b, thread_size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

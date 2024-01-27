#include "CudaNeuralNetwork.hpp"
#include "CNNHelper.hpp"
#include "NeuralNetwork.hpp"
#include "kernel.h"
#include <stdio.h>
#include "ImageData.hpp"
#include <map>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd)
{
    if (nnd.nb_input_layer <= 0 || nnd.nb_col_hiden_layer <= 0 || nnd.nb_hiden_layer <= 0 || nnd.nb_output_layer <= 0)
    {
        fprintf(stderr, "createNeuralNetwork failed! invalid input NeuralNetworkData\n");
        return nullptr;
    }

    return new NeuralNetwork(nnd);
}

std::map<const std::string, std::vector<float*>> charger(const std::string& nom_fichier, int* size)
{
    std::map<const std::string, std::vector<float*>> donnees;
    std::ifstream fichier(nom_fichier, std::ios::binary);
    if (fichier.is_open())
    {
        // Lecture de la taille de la map
        size_t taille_map;
        size_t image_size;
        fichier.read(reinterpret_cast<char*>(&taille_map), sizeof(size_t));
        fichier.read(reinterpret_cast<char*>(&image_size), sizeof(size_t));
        *size = image_size;
        // Lecture des données pour chaque paire clé-valeur dans la map
        for (size_t i = 0; i < taille_map; ++i) {
            // Lecture de la taille de la clé
            size_t taille_cle;
            fichier.read(reinterpret_cast<char*>(&taille_cle), sizeof(size_t));
            // Lecture de la clé
            char* cle = new char[taille_cle + 1];
            fichier.read(cle, taille_cle);
            cle[taille_cle] = '\0';

            // Lecture de la taille du vecteur
            size_t taille_vecteur;
            fichier.read(reinterpret_cast<char*>(&taille_vecteur), sizeof(size_t));

            // Lecture des données float du vecteur
            std::vector<float*> vecteur;
            for (size_t j = 0; j < taille_vecteur; ++j) {
                float* valeur = new float[image_size * image_size * 3];
                fichier.read(reinterpret_cast<char*>(valeur), sizeof(float) * image_size * image_size * 3);
                vecteur.push_back(valeur);
            }

            // Stockage des données dans la map
            donnees[std::string(cle)] = vecteur;
            delete[] cle;
        }
        fichier.close();
        std::cout << "Données chargées avec succès depuis " << nom_fichier << std::endl;
    }
    else {
        std::cerr << "Impossible d'ouvrir le fichier pour chargement." << std::endl;
    }
    return donnees;
}

void trainingNeuralNetwork(NeuralNetwork* nn, const std::string& dataSetPath, float min_percent_error_train)
{
    int size;
    std::map<const std::string, std::vector<float*>> data = charger(dataSetPath,&size);
    nn->trainingDataSet(data, size, min_percent_error_train);
}

void trainingNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<float>> input, const std::vector<std::vector<float>> output, float min_percent_error_train)
{
    nn->trainingInput(input, output, min_percent_error_train);
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

void sauvegarder(const std::map<const std::string, std::vector<float*>>& data, const std::string& nom_fichier, size_t image_size)
{
    std::ofstream fichier(nom_fichier, std::ios::binary);
    if (fichier.is_open()) 
    {
        // Écriture de la taille de la map
        size_t taille_map = data.size();
        fichier.write(reinterpret_cast<const char*>(&taille_map), sizeof(size_t));
        // Taille d'une image 
        fichier.write(reinterpret_cast<const char*>(&image_size), sizeof(size_t));
        // Pour chaque paire clé-valeur dans la map
        for (const auto& paire : data) 
        {
            // Écriture de la taille de la clé
            size_t taille_cle = paire.first.size();
            fichier.write(reinterpret_cast<const char*>(&taille_cle), sizeof(size_t));
            // Écriture de la clé
            fichier.write(paire.first.c_str(), taille_cle);

            // Écriture de la taille du vecteur
            size_t taille_vecteur = paire.second.size();
            fichier.write(reinterpret_cast<const char*>(&taille_vecteur), sizeof(size_t));

            // Écriture des données float du vecteur
            for (size_t i = 0; i < taille_vecteur; ++i) 
            {
                fichier.write(reinterpret_cast<const char*>(paire.second[i]), sizeof(float)* image_size * image_size *3);
            }
        }
        fichier.close();
        std::cout << "Données sauvegardées avec succès dans " << nom_fichier << std::endl;
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
    std::map<const std::string, std::vector<float*>> data;
    for (const auto& pair : m_map_dataset) 
    {        
        std::vector<float*> d;
        for (const auto& value : pair.second) 
        {
            float* col = new float[image_data_size * image_data_size * 3];

            int offset = 0;
            int offset2 = 0;
            if (value.ch == 3)
            {
                for (int i = 0; i < value.height; i++)
                {
                    for (int j = 0; j < value.width; j++)
                    {
                        offset = (i * value.width + j) * value.ch;
                        col[offset] = (((float)value.data[offset] / 255.0f) * 2.0f) - 1.0f;
                        col[offset + 1] = (((float)value.data[offset + 1] / 255.0f) * 2.0f) - 1.0f;
                        col[offset + 2] = (((float)value.data[offset + 2] / 255.0f) * 2.0f) - 1.0f;
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
                        col[offset2] = (((float)value.data[offset] / 255.0f) * 2.0f) - 1.0f;
                        col[offset2 + 1] = (((float)value.data[offset + 1] / 255.0f) * 2.0f) - 1.0f;
                        col[offset2 + 2] = (((float)value.data[offset + 2] / 255.0f) * 2.0f) - 1.0f;
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

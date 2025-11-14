#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Bug del syncthreads
#if defined(_MSC_VER) && !defined(__CUDACC__)
inline void __syncthreads() {  }
#endif

#include <vector>
#include <string>
#include <thread>
#include <numeric>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>
#include <cstdlib>
#include <mutex>

using namespace std;

// Variables varias

int IMAGEN = 3072;
int CLASES = 10;
int EPOCAS = 15;

// PERCEPTRON MEJORADO! XD

struct Neuron {
    vector<float> pesos;
    Neuron() {
        pesos.resize(IMAGEN + 1);
        // NO PONERLE 0'S!, SINO NUNCA APRENDE NADA XD
        // Pesos randoms
        random_device rd;
        mt19937 generator(rd());
        normal_distribution<float> distribution(0.0f, 0.01f);

        for (float& i : pesos) {
            i = distribution(generator);
        }
    }

    // El output
    float output(float* n) { // Basicamente aqui calculamos el producto escalar
        float sum = pesos[0];
        for (int i = 0; i < IMAGEN; ++i) {
            sum = sum + pesos[i + 1] * n[i];
        }
        return sum;
    }
};

int predict_cuda(const float* h_x);

struct Perceptron {
    vector<Neuron> NEURONAS = vector<Neuron>(CLASES);
    float TasaAprendizaje = 0;

    Perceptron(float lr) {
        TasaAprendizaje = lr;
    }

    
    int Prediccion(float* n) {
        int clase = 0;
        float output = NEURONAS[0].output(n);

        // Es solo hacer un for revisando a cada cual "Pertenece mas" xd
        for (int i = 1; i < CLASES; ++i) {
            float TRA = NEURONAS[i].output(n);
            if (output < TRA) {
                output = TRA; // ES SOLO HACER UN FOR
                clase = i; // EUREKA!
            }
        }

        return clase; // Devolvemos la mejor
    }

    void Entrenamiento(float* n, int k) {
        // Este entrenamiento es por la regla de Rossenblat
        // ADAPTADO
        //vector<float> OUTPUTS(CLASES);

        int prediccion = Prediccion(n);
        if (prediccion == k) {
            return;
        }

        NEURONAS[k].pesos[0] = NEURONAS[k].pesos[0] + TasaAprendizaje;
        NEURONAS[prediccion].pesos[0] = NEURONAS[prediccion].pesos[0] - TasaAprendizaje;

        for (int j = 0; j < IMAGEN; ++j) {
            float delta = TasaAprendizaje * n[j];
            NEURONAS[k].pesos[j + 1] = NEURONAS[k].pesos[j + 1] + delta;
            NEURONAS[prediccion].pesos[j + 1] = NEURONAS[prediccion].pesos[j + 1] - delta;
        }
    }

};

string PATH = "C:/Users/LENOVO/Desktop/7. Inteligencia Artifical/Tareas/Perceptron IA/ProyectoIAPerceptron/ProyectoPerceptron/ProyectoPerceptron";

struct Imagen {
    int clase = 0;
    int pixeles[3072];
};

void load(vector<Imagen>& ENTRENAMIENTO, vector<Imagen>& TESTING) {

    for (int i = 1; i <= 5; ++i) {
        string archivo = PATH + "/data_batch_" + to_string(i) + ".bin";
        ifstream f(archivo, ios::binary);

        for (int img = 0; img < 10000; ++img) {
            unsigned char clase;
            unsigned char pixeles[3072];

            f.read(reinterpret_cast<char*>(&clase), 1);
            f.read(reinterpret_cast<char*>(pixeles), 3072);

            Imagen IMG;
            IMG.clase = static_cast<int>(clase);

            for (int k = 0; k < 3072; ++k) {
                IMG.pixeles[k] = static_cast<int>(pixeles[k]);
            }

            ENTRENAMIENTO.push_back(IMG);
        }
    }

    string testfile = PATH + "/test_batch.bin";
    ifstream ftest(testfile, ios::binary);

    for (int i = 0; i < 10000; ++i) {
        unsigned char clase;
        unsigned char pixeles[3072];

        ftest.read(reinterpret_cast<char*>(&clase), 1);
        ftest.read(reinterpret_cast<char*>(pixeles), 3072);

        Imagen IMG;
        IMG.clase = static_cast<int>(clase);

        for (int k = 0; k < 3072; ++k) {
            IMG.pixeles[k] = static_cast<int>(pixeles[k]);
        }

        TESTING.push_back(IMG);
    }
}

float meanR = 0.4914f;
float meanG = 0.4822f;
float meanB = 0.4465f;

float stdR = 0.2470f;
float stdG = 0.2435f;
float stdB = 0.2616f;


void Normalizar(int* n, float* x) {
    for (int i = 0; i < 1024; ++i) {
        float r = static_cast<unsigned char>(n[i]) / 255.0f;
        float g = static_cast<unsigned char>(n[i + 1024]) / 255.0f;
        float b = static_cast<unsigned char>(n[i + 2048]) / 255.0f;
        x[i] = (r - meanR) / stdR;   // R
        x[i + 1024] = (g - meanG) / stdG;  // G
        x[i + 2048] = (b - meanB) / stdB;  // B
    }
}

// PARALELIZADO

// CUDA


float* d_W = 0;
float* d_bias = 0;
float* d_x = 0;
float* d_scores = 0;

bool init_device_buffers(const Perceptron& M) {

    // Empacar W (sin bias) y bias
    vector<float> h_W((size_t)CLASES * IMAGEN);
    vector<float> h_bias(CLASES);

    for (int c = 0; c < CLASES; ++c) {

        h_bias[c] = M.NEURONAS[c].pesos[0];

        for (int j = 0; j < IMAGEN; ++j) {

            h_W[(size_t)c * IMAGEN + j] = M.NEURONAS[c].pesos[j + 1];

        }
    }

    // Se deben crear espacios en la tarjeta a modo de C
    cudaMalloc(&d_W, (size_t)CLASES * IMAGEN * sizeof(float));
    cudaMalloc(&d_bias, (size_t)CLASES * sizeof(float));
    cudaMalloc(&d_x, (size_t)IMAGEN * sizeof(float));
    cudaMalloc(&d_scores, (size_t)CLASES * sizeof(float));

    cudaMemcpy(d_W, h_W.data(), (size_t)CLASES * IMAGEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), (size_t)CLASES * sizeof(float), cudaMemcpyHostToDevice);

    return 1;
}

__global__ void Producto(const float* __restrict__ W, const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ scores, int IMAGEN) {
    extern __shared__ float sh[]; // reduccion por bloque
    int c = blockIdx.x;   // esto procesa el bloque o eso creo
    int thread = threadIdx.x;
    float TRA = 0.f;

    // Sumatoria

    // REGLA DE PROPAGACION DEL PERCEPTRON

    for (int j = thread; j < IMAGEN; j = j + blockDim.x) {
        TRA = TRA + W[c * IMAGEN + j] * x[j];
    }
    sh[thread] = TRA;


    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (thread < s) {
            sh[thread] =sh[thread] + sh[thread + s];
        }
        __syncthreads();
    }

    if (thread == 0) {
        scores[c] = sh[0] + bias[c];
    }
}

// Actualuzar pesos en CPU (Rosenblatt)
void ActualizaPesos(const Perceptron& M) {

    vector<float> h_W((size_t)CLASES * IMAGEN);
    vector<float> h_bias(CLASES);

    for (int i = 0; i < CLASES; ++i) {

        h_bias[i] = M.NEURONAS[i].pesos[0];

        for (int j = 0; j < IMAGEN; ++j) {

            h_W[(size_t)i * IMAGEN + j] = M.NEURONAS[i].pesos[j + 1];

        }
            
    }
    cudaMemcpy(d_W, h_W.data(), (size_t)CLASES * IMAGEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), (size_t)CLASES * sizeof(float), cudaMemcpyHostToDevice);
}

// PREDICCION (CUDA)

int predict_cuda(const float* h_x) {

    cudaMemcpy(d_x, h_x, (size_t)IMAGEN * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(CLASES);
    dim3 block(256);
    size_t shmem = block.x * sizeof(float);

    Producto << < grid, block, shmem >> > (d_W, d_x, d_bias, d_scores, IMAGEN);
    cudaDeviceSynchronize();

    vector<float> h_scores(CLASES);
    cudaMemcpy(h_scores.data(), d_scores, (size_t)CLASES * sizeof(float), cudaMemcpyDeviceToHost);

    int best = 0;
    float bestv = h_scores[0];

    for (int i = 1; i < CLASES; ++i) {

        if (h_scores[i] > bestv) { // Solo es seleccionar al mejor

            bestv = h_scores[i];
            best = i;

        }
    }
    return best;
}



int main() {

    vector<Imagen> ENTRENAMIENTO;
    vector<Imagen> TESTING;

    load(ENTRENAMIENTO, TESTING);

    mt19937 rng(random_device{}());

    Perceptron modelo(0.1f); // tasa de aprendizaje
    vector<float> x(IMAGEN);

    init_device_buffers(modelo);

    

    // ENTRENAMIENTO
    for (int ep = 1; ep <= EPOCAS; ++ep) {

        shuffle(ENTRENAMIENTO.begin(), ENTRENAMIENTO.end(), rng);
        cout << "Epoca: " << ep << endl;

        int aciertos = 0;




        for (auto i = 0; i < ENTRENAMIENTO.size(); ++i) {

            auto& s = ENTRENAMIENTO[i];

            Normalizar(s.pixeles, x.data());

            // ROSENBLAT!

            // Normal
            //int pred = modelo.Prediccion(x.data());

            //CUDA
            int pred = predict_cuda(x.data());

            aciertos = aciertos + (pred == s.clase);

            modelo.Entrenamiento(x.data(), s.clase);
            //aciertos = aciertos + (modelo.Prediccion(x.data()) == s.clase);
        }

        //cout << "wee wooo wee wooo" << endl;
        float precision = 100.0 * aciertos / ENTRENAMIENTO.size();
        cout << "Entrenamiento " << precision << endl;

        // CUDA

        ActualizaPesos(modelo);

        // TESTING

        precision = 0;
        for (const auto& s : TESTING) {

            Normalizar(const_cast<int*>(s.pixeles), x.data());

            // Normal
            //precision = precision + (modelo.Prediccion(x.data()) == s.clase);

            // CUDA
            precision = precision + (predict_cuda(x.data()) == s.clase);


        }

        precision = precision * 100.0 / TESTING.size();
        cout << "Testing " << precision << endl << endl;

    }

    cout << "Completado" << endl;
    return 0;
}

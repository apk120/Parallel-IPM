#include <bits/stdc++.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>

using namespace std;
#define CUDIE(result) {\
        cudaError_t e = (result);\
        if (e != cudaSuccess) {\
            std::cerr << __FILE__ << ":" << __LINE__;\
            std::cerr << " CUDA Runtime Error: " << cudaGetErrorString(e) << "\n";\
            exit((int)e);\
        }}

#define CUDIE0() CUDIE(cudaGetLastError())

#define THREADS_PER_BLOCK 256

void createA(float *A, float *csrValA, int *csrRowA, int *csrColA, int m, int n, int nnz)
{
    for (int i  =0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i + m*j] = 0;
    int k = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowA[i] - 1; j < csrRowA[i+1] - 1; j++)
        {
            A[i + m*csrColA[k]] = csrValA[k];
            k++;
        }
    }
}

void printVectorfloat(const float *V, int m)
{
    for (int i = 0; i < m; i++)
        std::cout << V[i] << " ";
    std::cout << std::endl;
}

int main()
{
    float A[8], csrValA[4];
    int csrRowA[3], csrColA[4];
    int m = 2, n = 4, nnz = 4;
    csrRowA[0] = 1, csrRowA[1] = 3, csrRowA[2] = 5;
    csrColA[0] = 0, csrColA[1] = 1, csrColA[2] = 2, csrColA[3] = 3;
    csrValA[0] = 1, csrValA[1] = 4, csrValA[2] = 2, csrValA[3] = 16;
    createA(A, csrValA, csrRowA, csrColA, m, n, nnz);
    printVectorfloat(A, 8);
}
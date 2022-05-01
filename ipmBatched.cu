#include <iostream>
#include <sstream>
#include <assert.h>
#include <cuda.h>
// #include <cusolverSp.h>
#include <cuda_runtime_api.h>
// #include <cusolverRf.h>
#include <cuda_runtime.h>
// #include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include "inc/helper_string.h"
#include "inc/helper_cusolver.h"
#include "inc/helper_cuda.h"
// #include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cusolverDn.h>
#include "cusolver_utils.h"
using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define CUDIE(result) {\
        cudaError_t e = (result);\
        if (e != cudaSuccess) {\
            std::cerr << __FILE__ << ":" << __LINE__;\
            std::cerr << " CUDA Runtime Error: " << cudaGetErrorString(e) << "\n";\
            exit((int)e);\
        }}

#define CUDIE0() CUDIE(cudaGetLastError())


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define THREADS_PER_BLOCK 256
using data_type = double;

void printVectordouble(const double *V, int m);
// void sparesify(double *M, int m, int n, double *A, int *IA, int *JA);
int findNNZ(const double *M, int N);
void printMatrix(const double *A, int nr_rows_A, int nr_cols_A);
void printVector(const int *V, int m);
void getMatSize(string fname, int *rows, int *nnz);
void loadMat(double *A, int *IA, int *JA, string fname, int *rows, int *nnz);
void loadVec(double *V, string fname);
void writeVec(double *V, int n, string fname);
void writeVecInt(int *V, int n, string fname);
void createA(double *A, double *csrValA, int *csrRowA, int *csrColA, int m, int n, int nnz, int baseA);
int solve (int rowsA, int colsA, int N, int nnzA, int batchSize, int baseA, \
            double *h_csrValA, int *h_csrRowPtrA, int *h_csrColIndA, \
             double *x, double *b, struct testOpts opts);
int xsc(int *hA_csrOffsets, int *hA_columns, double *hA_values,int *hB_csrOffsets, int *hB_columns, double *hB_values, int *hC_csrOffsets, int *hC_columns, double *hC_values, int C_nnz, double *resC) ;

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts);
/*
* Batched Dot product of 2 Dense Vectors
* Each Block works on a single batch of vectors
* Number of Blocks must be >= Number of Batches (no_batch) 
*/
__global__ void dot(double *x_b, double *s_b, double *mu, int n, int no_batch)
{
    __shared__ double temp[THREADS_PER_BLOCK];
    temp[threadIdx.x] = 0.0;
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    double *x = &x_b[batch* n];
    double *s = &s_b[batch * n];   
    
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        temp[i%THREADS_PER_BLOCK] += x[i] * s[i];
    __syncthreads();

    if (threadIdx.x == 0)
    {
        mu[batch] = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
            mu[batch] += temp[i]/n;
    }
}

/*
* Multiplication of Diagonal Matrix with vector 
* Diagonal Matrix is input as a Dense Vector of Diagonal Elements
* type = 0 for d1*x/d2, type = 1 for x/d2
*/
__global__ void diag_vector_mul(double *diag_b1, double *diag_b2, double *x_b, double *res_b, \
                                int n, int type, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else 
        return;
    
    double *diag1 = &diag_b1[n * batch];
    double *diag2 = &diag_b2[n * batch];
    double *x = &x_b[n * batch];
    double *res = &res_b[n * batch];
    for(int i = threadIdx.x; i < n; i += blockDim.x)
    {
        double di;
        double sign = 1.0;
        if (diag2[i] < 0)
            sign = -1.0;
            
        if (abs(diag2[i]) < 1e-9)
            di = 1e-9*sign;
        else
            di = diag2[i];

        if (type == 0)
            res[i] = diag1[i] * x[i] / di;
        else    
            res[i] = x[i] / di;
    }
}
/*
* Batched Vector Addition of 3 vectors at a time
* Each Block works on a single batch of vectors
* Number of Blocks must be >= Number of Batches (no_batch)
*/
__global__ void vector_add(double *x_b, double *y_b, double *z_b, double *res_b, double a1, double a2, \
                        double a3, int n, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else    
        return;
    double *x = &x_b[batch * n];
    double *y = &y_b[batch * n];
    double *z = &z_b[batch * n];
    double *res = &res_b[batch * n];

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        res[i] = a1 * x[i] + a2 * y[i] + a3 * z[i];
}

/*
* Batched Vector Element Wise Multiplication of 
* 2 vectors-> res[i] = a1*mu + a2*x[i]*y[i] 
* Each Block works on a single batch of vectors
* Number of Blocks must be >= Number of Batches (no_batch)
*/
__global__ void vector_mul(double *x_b, double *y_b, double *mu, double *res_b,\
                            double a1, double a2, int n, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    double *x = &x_b[batch * n];
    double *y = &y_b[batch * n];
    double *res = &res_b[batch * n];

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        res[i] = a1 * mu[batch] + a2 * x[i] * y[i]; 
}

/*
*
*/
__global__ void find_update_param(double *x_b, double *s_b, double *ap_aff, \
                    double *ad_aff, double *dx_aff_b, double *ds_aff_b, int n, int no_batch)
{
    int batch;
    if(blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;

    if (threadIdx.x == 0)
    {
        double *x = &x_b[n * batch];
        double *dx_aff = &dx_aff_b[n * batch];
        double *s = &s_b[n * batch];
        double *ds_aff = &ds_aff_b[n * batch];
        for (int i = 0; i < n; i++)
        {
            if (dx_aff[i] < 0)
                ap_aff[batch] = min(ap_aff[batch], -0.9*x[i]/dx_aff[i]);
            if (ds_aff[i] < 0)
                ad_aff[batch] = min(ad_aff[batch], -0.9*s[i]/ds_aff[i]);
        }
    }
}

/*
*
*/
__global__ void update_vars(double *x_b, double *s_b, double *y_b, double *dx_aff_b,\
            double *ds_aff_b, double *dy_aff_b, double *ap_aff, double *ad_aff, \
            int m, int n, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else    
        return;
    
    double *x = &x_b[n * batch];
    double *y = &y_b[m * batch];
    double *s = &s_b[n * batch];
    double *dx_aff = &dx_aff_b[n * batch];
    double *dy_aff = &dy_aff_b[m * batch];
    double *ds_aff = &ds_aff_b[n * batch];

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        x[i] = x[i] + ap_aff[batch] * dx_aff[i];
        s[i] = s[i] + ad_aff[batch] * ds_aff[i];
        if (i < m)
            y[i] = y[i] + ad_aff[batch] * dy_aff[i];
    }
}

/*
*   if op == 0
        AD = A * X
    else
        AD = A * X / Y
*/
__global__ void diag_matmul(int m, int n, int no_batch, int op, double *A_batched, double *x_batched, double *y_batched, double *AD_batched, int baseA)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    
    double *A = &A_batched[batch * m * n];
    double *x = &x_batched[batch * n];
    double *y = &y_batched[batch * n];
    double *AD = &AD_batched[batch * m * n];

    for (int i = threadIdx.x; i < m; i += blockDim.x)
    {
        for (int j = 0; j < n; j++)
        {
            if (op == 0)
            {
                AD[i * n + j] = A[i * n + j] * x[j];
            }
            else if (op == 1)
            {
                double sign = 1.0, zy = 1.0;
                if (y[j] < 0)
                    sign = -1.0;
                if (abs(y[j]) < 1e-12)
                    zy = sign * 1e-12;
                else
                    zy = y[j];

                AD[i * n + j] = A[i * n + j] / zy;
            }
            else
            {
                double sign = 1.0, zy = 1.0;
                if (y[j] < 0)
                    sign = -1.0;
                if (abs(y[j]) < 1e-12)
                    zy = sign * 1e-12;
                else
                    zy = y[j];

                AD[i * n + j] = A[i * n + j] * x[j] / zy;
            }
        }
    }
}
void printM(int m, int n, double *A, int lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%lf ", A[i * n + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Batched Matrix Multiplication
 * 
 * @param m
 * @param k 
 * @param n  
 * @param batch_count 
 * @param lda - Leading dimension of A
 * @param ldb 
 * @param ldc 
 * @param A_batch - A is m * k matrix * number of batches 
 * @param B_batch - B is k * n matrix
 * @param C_batch - C is result (m * n matrix)
 */
void gemm_batched(int m, int k, int n, int batch_count, int lda, int ldb, int ldc, data_type *A_batch, data_type *B_batch, data_type *C_batch,\
                    cublasHandle_t cublasH, cudaStream_t stream)
{
    // cublasHandle_t cublasH = NULL;
    // cudaStream_t stream = NULL;

    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type **d_A_array = nullptr;
    data_type **d_B_array = nullptr;
    data_type **d_C_array = nullptr;

    std::vector<data_type *> d_A(batch_count, nullptr);
    std::vector<data_type *> d_B(batch_count, nullptr);
    std::vector<data_type *> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;


    // printf("A matrix\n");
    // for (int i = 0; i < batch_count; i++)
    //     printM(m, k, &A_batch[m * k * i], lda);
    // printf("-------\n");

    // printf("B matrix\n");
    // for (int i = 0; i < batch_count; i++)
    //     printM(k, n, &B_batch[k * n * i], ldb);
    // printf("-------\n");
    /* step 1: create cublas handle, bind a stream */
    // CUBLAS_CHECK(cublasCreate(&cublasH));

    // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    
    /* step 2: copy data to device */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(data_type) * m * k));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(data_type) * k * n));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * m * n));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * batch_count));

    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], &A_batch[i * m * k], sizeof(data_type) * m * k,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], &B_batch[i * k * n], sizeof(data_type) * k * n,
                                   cudaMemcpyHostToDevice, stream));
    }
    // for (int i = 0; i < batch_count; i++)
    // {
    //     d_A_array[i] = &A_batch[i * m * k];
    //     d_B_array[i] = &B_batch[i * k * n];
    // }
    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
                    

    /* step 3: compute */
    CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, n, m, k, &alpha, d_B_array, n,
                                    d_A_array, k, &beta, d_C_array, n, batch_count));
    
    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(&C_batch[i * m * n], d_C[i], sizeof(data_type) * m * n,
                                   cudaMemcpyDeviceToHost, stream));
    }
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // printf("C matrix\n");
    // for (int i = 0; i < batch_count; i++)
    //     printM(m, n, &C_batch[m * n * i], ldc);
    // printf("-------\n");
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    // CUBLAS_CHECK(cublasDestroy(cublasH));

    // CUDA_CHECK(cudaStreamDestroy(stream));

    // CUDA_CHECK(cudaDeviceReset());
}

/**
 * @brief Batched Matrix-vector multiplication
 * 
 * @param m 
 * @param k 
 * @param batch_count 
 * @param lda 
 * @param A_batch - A is m * k matrix * batch_count
 * @param B_batch - B is k * 1 vector * batch_count
 * @param C_batch - C is result m * 1 vector
 */
void gemv_batched(int m, int k, int batch_count, int lda, data_type *A_batch, data_type *B_batch, data_type *C_batch, \
                    cublasHandle_t cublasH, cudaStream_t stream)
{
    // cublasHandle_t cublasH = NULL;
    // cudaStream_t stream = NULL;

    const int n = 1;
    // const int lda = m;
    // const int ldb = k;
    // const int ldc = m;
    // const int batch_count = 2;

    /*
     *   A = | 1.0 | 2.0 | 5.0 | 6.0 |
     *       | 3.0 | 4.0 | 7.0 | 8.0 |
     *
     *   B = | 5.0 |  9.0 |
     *       | 7.0 | 11.0 |
     */

    // const std::vector<std::vector<data_type>> A_array = {{1.0 ,3.0, 2.0, 4.0},
    //                                                      {5.0, 7.0, 6.0, 8.0}};
    // const std::vector<std::vector<data_type>> B_array = {{5.0, 7.0},
    //                                                      {9.0, 11.0}};
    // std::vector<std::vector<data_type>> C_array(batch_count, std::vector<data_type>(m * n));

    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type **d_A_array = nullptr;
    data_type **d_B_array = nullptr;
    data_type **d_C_array = nullptr;

    std::vector<data_type *> d_A(batch_count, nullptr);
    std::vector<data_type *> d_B(batch_count, nullptr);
    std::vector<data_type *> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // printf("A[0]\n");
    // printM(m, k, &A_batch[0], lda);
    // printf("=====\n");

    // printf("A[1]\n");
    // printM(m, k, &A_batch[m * k], lda);
    // printf("=====\n");

    // printf("B[0]\n");
    // printM(k, n, &B_batch[0], ldb);
    // printf("=====\n");

    // printf("B[1]\n");
    // printM(k, n, &B_batch[k * n], ldb);
    // printf("=====\n");

    // /* step 1: create cublas handle, bind a stream */
    // CUBLAS_CHECK(cublasCreate(&cublasH));

    // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    /*Create Multiple streams*/
    // const int num_streams = 2;
    // cudaStream_t streams[num_streams];
    // for (int i = 0; i < num_streams; i++)
    // {
    //     CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    // }
    /* step 2: copy data to device */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(data_type) * m * k));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(data_type) * k * n));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * m * n));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * batch_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * batch_count));

    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], &A_batch[i * m * k], sizeof(data_type) *  m * k,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], &B_batch[i * k * n], sizeof(data_type) * k * n,
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_count,
                               cudaMemcpyHostToDevice, stream));

    cudaEventRecord(start);
    /* step 3: compute */
    CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, n, m, k, &alpha, d_B_array, n,
                                    d_A_array, k, &beta, d_C_array, n, batch_count));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float mili = 0;
    cudaEventElapsedTime(&mili, start, stop);
    cout << "Time GEMV: " << mili << "\n";
    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(&C_batch[i * m * n], d_C[i], sizeof(data_type) * m * n,
                                   cudaMemcpyDeviceToHost, stream));
    }

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
     *       | 43.0 | 50.0 | 151.0 | 166.0 |
     */
    // printf("C matrix\n");
    // for (int i = 0; i < batch_count; i++)
    //     printM(m, n, &C_batch[m * n * i], ldc);
    // printf("-------\n");
    // printf("C[0]\n");
    // print_vector(m, &C_batch[0]);
    // printf("=====\n");

    // printf("C[1]\n");
    // print_vector(m, &C_batch[m]);
    // printf("=====\n");
    
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }
    // for (int i = 0; i < num_streams; i++)
    // {
    //     CUDA_CHECK(cudaStreamDestroy(streams[i]));
    // }
    // CUBLAS_CHECK(cublasDestroy(cublasH));

    // CUDA_CHECK(cudaStreamDestroy(stream));

    // CUDA_CHECK(cudaDeviceReset());
}

void ports_batched(int m, int batchSize, data_type *A_batch, data_type *B_batch, data_type *C_batch)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    const int lda = m;
    const int ldb = m;
    const int nrhs = 1;

    std::vector<int> infoArray(batchSize, 0); /* host copy of error info */
    std::vector<double> L0(lda * m); /* cholesky factor of A0 */
    std::vector<double *> Aarray(batchSize, nullptr);
    std::vector<double *> Barray(batchSize, nullptr);

    double **d_Aarray = nullptr;
    double **d_Barray = nullptr;
    int *d_infoArray = nullptr;

    // std::printf("A0 = (matlab base-1)\n");
    // printM(m, m, &A_batch[0], lda);
    // std::printf("=====\n");

    // std::printf("A1 = (matlab base-1)\n");
    // printM(m, m, &A_batch[m * m], lda);
    // std::printf("=====\n");

    // std::printf("B0 = (matlab base-1)\n");
    // printM(m, 1, &B_batch[0], ldb);
    // std::printf("=====\n");
    // std::printf("B1 = (matlab base-1)\n");
    // printM(m, 1, &B_batch[m], ldb);
    // std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    /* step 2: copy A to device */
    for (int j = 0; j < batchSize; j++) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Aarray[j]), sizeof(double) * lda * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Barray[j]), sizeof(double) * ldb * nrhs));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * infoArray.size()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double *) * Aarray.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Barray), sizeof(double *) * Barray.size()));

    for (int j = 0; j < batchSize; j++)
    {
        CUDA_CHECK(cudaMemcpyAsync(Aarray[j], &A_batch[j * m * m], sizeof(double) * m * m,
                                cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(Barray[j], &B_batch[j * m], sizeof(double) * m, cudaMemcpyHostToDevice, stream));

    }

    CUDA_CHECK(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double) * Aarray.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Barray, Barray.data(), sizeof(double) * Barray.size(),
                               cudaMemcpyHostToDevice, stream));
    

    cudaEventRecord(start);
    /* step 3: Cholesky factorization */
    // CUSOLVER_CHECK(
    //     cusolverDnDpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_infoArray, batchSize));
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float mili = 0;
    // cudaEventElapsedTime(&mili, start, stop);
    // cout << "Time portf: " << mili << "\n";
    // CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int) * infoArray.size(),
    //                            cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaMemcpyAsync(L0.data(), Aarray[0], sizeof(double) * lda * m,
    //                            cudaMemcpyDeviceToHost, stream));

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // for (int j = 0; j < batchSize; j++) {
    //     std::printf("info[%d] = %d\n", j, infoArray[j]);
    // }

    // std::printf("L = (matlab base-1), upper triangle is don't care \n");
    // printM(m, m, L0.data(), lda);
    // std::printf("=====\n");


    /*
     * step 4: solve A0*X0 = B0
     *        | 1 |        | 10.5 |
     *   B0 = | 1 |,  X0 = | -2.5 |
     *        | 1 |        | -1.5 |
     */
    // cudaEventRecord(start);
    CUSOLVER_CHECK(cusolverDnDpotrsBatched(cusolverH, uplo, m, nrhs, /* only support rhs = 1*/
                                           d_Aarray, lda, d_Barray, ldb, d_infoArray, batchSize));

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&mili, start, stop);
    // cout << "Time ports: " << mili << "\n";
    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    for (int i = 0; i < batchSize; i++)
    {
        CUDA_CHECK(
        cudaMemcpyAsync(&C_batch[i * m], Barray[i], sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    }
    

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::printf("after potrsBatched: infoArray[0] = %d\n", infoArray[0]);
    // if (0 > infoArray[0]) {
    //     std::printf("%d-th parameter is wrong \n", -infoArray[0]);
    //     exit(1);
    // }

    // std::printf("X0 = (matlab base-1)\n");
    // printM(m, 1, &C_batch[0], ldb);
    // std::printf("=====\n");

    // std::printf("X1 = (matlab base-1)\n");
    // printM(m, 1, &C_batch[m], ldb);
    // std::printf("=====\n");
    /* free resources */
    CUDA_CHECK(cudaFree(d_Aarray));
    CUDA_CHECK(cudaFree(d_Barray));
    CUDA_CHECK(cudaFree(d_infoArray));
    for (int j = 0; j < batchSize; j++) {
        CUDA_CHECK(cudaFree(Aarray[j]));
        CUDA_CHECK(cudaFree(Barray[j]));
    }

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    // CUDA_CHECK(cudaDeviceReset());

}


__global__ void multiplyAV(int m, int n, int batch_count, double *A_batch, double *v_batch, double *res_batch)
{
    int batch;
    if (blockIdx.x < batch_count)
        batch = blockIdx.x;
    else
        return;
    
    double *A = &A_batch[batch * m * n];
    double *v = &v_batch[batch* n];
    double *res = &res_batch[batch * m];

    for (int i = threadIdx.x; i < m; i += blockDim.x)
    {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += A[i * n + j] * v[j];
        }
        res[i] = sum;
    }
}


int main(int argc, char *argv[])
{
    int m = 200, n = 400, batches = 1000, iter = 2;//, nnz 
    double *A, *ADA, *AS, *AD, *AT;//*A
    double *x, *y, *s, *c, *b, *dx_aff, *dy_aff, *ds_aff, *rd, *rp, *rc, *v;   
    double *d_i1, *d_i2, *d_i3, *d_i4;
    //double sigma = 0.8;
    double *ap_aff, *ad_aff, *mu_aff, *mu, *cost;
    int base = 0;
    // struct testOpts opts;
    double start_t, stop_t, time_preop, time_ada, time_post, time_gemv;

    // parseCommandLineArguments(argc, argv, opts);
    // findCudaDevice(argc, (const char **)argv);


    cudaMallocManaged(&A, sizeof(double) * m * n * batches);
    cudaMallocManaged(&AS, sizeof(double) * m * n * batches);
    cudaMallocManaged(&AD, sizeof(double) * m * n * batches);
    cudaMallocManaged(&AT, sizeof(double) * m * n * batches);
    cudaMallocManaged(&ADA, sizeof(double) * m * m * batches);
    CUDIE0();
    cudaMallocManaged(&x, sizeof(double)* n * batches);
    cudaMallocManaged(&s, sizeof(double)* n * batches);
    cudaMallocManaged(&y, sizeof(double)* m * batches);
    cudaMallocManaged(&c, sizeof(double)* n * batches);
    cudaMallocManaged(&b, sizeof(double)* m * batches);
    cudaMallocManaged(&dx_aff, sizeof(double) * n *batches);
    cudaMallocManaged(&dy_aff, sizeof(double) * m * batches);
    cudaMallocManaged(&d_i1, sizeof(double) * m * batches);
    cudaMallocManaged(&d_i2, sizeof(double) * m * batches);
    cudaMallocManaged(&d_i3, sizeof(double) * m * batches);
    cudaMallocManaged(&ds_aff, sizeof(double) * n * batches);
    cudaMallocManaged(&d_i4, sizeof(double) * n * batches);
    cudaMallocManaged(&rd, sizeof(double)* n * batches);
    cudaMallocManaged(&rp, sizeof(double)* m * batches);
    cudaMallocManaged(&rc, sizeof(double)* n * batches);
    cudaMallocManaged(&v, sizeof(double)* n * batches);
    cudaMallocManaged(&ap_aff, sizeof(double) * batches);
    cudaMallocManaged(&ad_aff, sizeof(double) * batches);
    cudaMallocManaged(&mu_aff, sizeof(double) * batches);
    cudaMallocManaged(&mu, sizeof(double) * batches);
    cudaMallocManaged(&cost, sizeof(double) * batches);
    CUDIE0();

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    /*Create Multiple streams*/
    const int num_streams = 3;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    /*Initialization of the Problem*/
    double *ib, *ic, *ix, *is, *iy;
    ib = (double *)malloc(m * sizeof(double));
    ic = (double *)malloc(n * sizeof(double));
    ix = (double *)malloc(n * sizeof(double));
    is = (double *)malloc(n * sizeof(double));
    iy = (double *)malloc(m * sizeof(double));
    loadVec(ib, "B_r_dense.csv");
    loadVec(ic, "C_r_dense.csv");
    loadVec(ix, "x_ini_dense.csv");
    loadVec(is, "s_ini_dense.csv");
    loadVec(iy, "y_ini_dense.csv");
    // double ib[2] = {0.2964, 0.3061}, ic[3] = {0.64, 0.84, 0.38};
    // double ix[3] = {0.436, 0.596, 0.554}, is[3] = {0.7355, 0.5293, 0.2713}, iy[2] = {1.08, 1.1877};
    
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < m; j++) {
            b[m*i + j] = ib[j];
            y[m*i + j] = iy[j];
        }
    }  
    
    // cout << miny << " " << maxy << endl;      
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < n; j++) {
            c[n*i + j] = ic[j];
            x[n*i + j] = ix[j];
            s[n*i + j] = is[j];
        }
    }

    /*printVector(csrColAT, 10);
    printVectordouble(csrValAT, 10);
    printVector(csrRowAT, 10);
    writeVecInt(csrColAT, nnz, "csrColAt_bef.txt");
    writeVecInt(csrRowAT, n+1, "csrRowAt_first.txt");*/
    //writeVec(csrValAT, 1000, "csrValAt_bef.txt");
    //sparesify(A, n, m, csrValAT, csrRowAT, csrColAT);


    // A[0] = 0.06626872; A[1] = 0.49782317; A[2] = 0.16945034; 
    // A[3] = 0.27174321; A[4] = 0.22702761; A[5] = 0.35243303;
    // AT[0] = 0.06626872; AT[1] = 0.27174321;
    // AT[2] = 0.49782317; AT[3] = 0.22702761;
    // AT[4] = 0.16945034; AT[5] = 0.35243303;
    loadVec(A, "A_r_dense.csv");
    loadVec(AT, "A_r_t_dense.csv");
    for (int i = 1; i < batches; i++)
    {
        for (int j = 0; j < m * n; j++)
        {
            A[i * m * n + j] = A[j];
            AT[i * m * n + j] = AT[j];
        }
    }   


    /*---Initialization*/

    //printVectordouble(x, n*batches);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;
    cudaEventRecord(start);

    /*Main Loop for IPM iterations*/
    iter = 10;
    for (int i = 0; i < iter; i++)
    {
        start_t = second();
        start_t = second();
        dot<<<batches, THREADS_PER_BLOCK, 0, streams[0]>>>(x, s, mu, n, batches);
        // cudaDeviceSynchronize();
        // CUDIE0();
        // cout << "MU" << endl;
        // for (int j = 0; j < min(batches, 10); j++)
        //     cout << mu[j]*n << " " << n << " ";
        // cout << endl;
        //writeVecInt(csrRowAT, n+1, "csrRowAt_bef.txt");
        // csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, y, rd,\
        //                                             n, m, nnz, batches, base);

        multiplyAV<<<batches, THREADS_PER_BLOCK>>>(n, m, batches, AT, y, rd);
        // gemv_batched(n, m, batches, n, AT, y, rd, cublasH, stream);
        cudaDeviceSynchronize();
        CUDIE0();
        /*cout << "xxxxxx" << endl;
        printVector(csrColAT, 10);
        printVectordouble(csrValAT, 10);
        printVector(csrRowAT, 10);
        printVectordouble(rd, 10);*/
        /*writeVecInt(csrColAT, 1000, "csrColAt.txt");
        writeVecInt(csrRowAT, n+1, "csrRowAt.txt");
        writeVec(csrValAT, 1000, "csrValAt.txt");*/
        //writeVec(rd, n, "yuss2.txt");
        
        vector_add<<<batches, THREADS_PER_BLOCK, 0, streams[0]>>>(c, s, rd, rd, 1, -1, -1, n, batches);
        // cudaDeviceSynchronize();
        // CUDIE0();
        // writeVec(rd, n, "yuss2.txt");
        
        // csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA, x, rp,\
        //                                             m, n, nnz, batches, base);

        multiplyAV<<<batches, THREADS_PER_BLOCK>>>(m, n, batches, A, x, rp);
        // gemv_batched(m, n, batches, n, A, x, rp, cublasH, stream);
        // cudaDeviceSynchronize();
        // CUDIE0();
        vector_add<<<batches, THREADS_PER_BLOCK, 0, streams[0]>>>(b, rp, rp, rp, 1, -1, 0, m, batches);
        // cudaDeviceSynchronize();
        // CUDIE0();
        //writeVec(rp, m, "yuss1.txt");
        vector_mul<<<batches, THREADS_PER_BLOCK, 0, streams[1]>>>(x, s, mu, rc, 0, -1, n, batches);
        // cudaDeviceSynchronize();
        // CUDIE0();
        
        // csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
        //                          csrValAS, s, s, 1, m, n, nnz, batches, base);
        diag_matmul<<<batches, THREADS_PER_BLOCK, 0, streams[2]>>>(m, n, batches, 1, A, s, s, AS, base); 
        
        cudaDeviceSynchronize();
        CUDIE0();
        // csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAS, csrRowA, csrColA, rc, d_i1,\
        //                                             m, n, nnz, batches, base);

        multiplyAV<<<batches, THREADS_PER_BLOCK>>>(m, n, batches, AS, rc, d_i1);
        // gemv_batched(m, n, batches, m, AS, rc, d_i1, cublasH, stream);


        // cudaDeviceSynchronize();
        // CUDIE0();
        // csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
        //                          csrValAD, x, s, 2, m, n, nnz, batches, base);
        diag_matmul<<<batches, THREADS_PER_BLOCK>>>(m, n, batches, 2, A, x, s, AD, base);
        cudaDeviceSynchronize();
        CUDIE0();

        stop_t = second();
        time_preop = stop_t - start_t;

        // cout << "ADAT start_t\n";
        start_t = second();
        start_t = second();
        
        // ADAT<<<batches, THREADS_PER_BLOCK>>>(A, csrValAD, csrRowA, csrColA, csrValADA,\
        //                        csrRowADA, csrColADA, x, s, m, n, nnz, nnzAAt, batches, base);
        gemm_batched(m, n, m, batches, m, n, m, AD, AT, ADA, cublasH, stream); //wrong
        // int crr = 0;
        // cout << "ADAT start_t\n";
        // int ccmn(int *hA_csrOffsets, int *hA_columns, double *hA_values,int *hB_csrOffsets, int *hB_columns, double *hB_values, int *hC_csrOffsets, int *hC_columns, double *hC_values, int C_nnz, double *resC) 
        // crr = xsc(csrRowA, csrColA, csrValAD, csrRowAT, csrColAT, csrValAT, csrRowADA, csrColADA, csrValAAT, nnzAAt, csrValADA);
        //return 0;
        // if (crr != 0)
        // {
        //     printf("Error\n");
        //     return 0;
        // }
        // cudaDeviceSynchronize();
        // CUDIE0();
        stop_t = second();
        time_ada = stop_t - start_t;

        start_t = second();
        start_t = second();
        /*writeVec(csrValADA, nnzAAt, "csrValADA.txt");
        writeVecInt(csrColADA, nnzAAt, "csrColADA.txt");
        writeVecInt(csrRowADA, m + 1, "csrRowADA.txt");*/
        // cout << "ADAT End\n";
        //return 0;
        // csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAD, csrRowA, csrColA, rd, d_i2,\
        //                                             m, n, nnz, batches, base);

        multiplyAV<<<batches, THREADS_PER_BLOCK>>>(m, n, batches, AD, rd, d_i2);
        // gemv_batched(m, n, batches, n, AD, rd, d_i2, cublasH, stream);

        cudaDeviceSynchronize();
        // CUDIE0();
        stop_t = second();
        time_gemv = stop_t - start_t;
        vector_add<<<batches, THREADS_PER_BLOCK>>>(d_i1, d_i2, rp, d_i3, -1, 1, 1, m, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        
        //writeVec(d_i2, m, "yuss3.txt");
        //writeVec(d_i3, m, "yuss.txt");
        // cout << m << " " << n << " " << " " << endl;
        //Solve for dy_aff
        // solve(m, m, m, nnzAAt, batches, base, csrValADA, csrRowADA, csrColADA, dy_aff, d_i3, opts);
        ports_batched(m, batches, ADA, d_i3, dy_aff);
        
        // csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, dy_aff, ds_aff,\
        //                                             n, m, nnz, batches, base);

        multiplyAV<<<batches, THREADS_PER_BLOCK>>>(n, m, batches, AT, dy_aff, ds_aff);
        // gemv_batched(n, m, batches, m, AT, dy_aff, ds_aff, cublasH, stream);

        // cudaDeviceSynchronize();
        vector_add<<<batches, THREADS_PER_BLOCK, 0, streams[0]>>>(ds_aff, rd, ds_aff, ds_aff, -1, 1, 0, n, batches);
        // cudaDeviceSynchronize();
        diag_vector_mul<<<batches, THREADS_PER_BLOCK, 0, streams[1]>>>(s, s, rc, d_i4, n, 1, batches);
        cudaDeviceSynchronize();
        diag_vector_mul<<<batches, THREADS_PER_BLOCK>>>(x, s, ds_aff, dx_aff, n, 0, batches);
        cudaDeviceSynchronize();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(dx_aff, d_i4, dx_aff, dx_aff, -1, 1, 0, n, batches);
        cudaDeviceSynchronize();

        for (int j = 0; j < batches; j++)
        {
            ap_aff[j] = 1.0;
            ad_aff[j] = 1.0;
        }
        find_update_param<<<batches, THREADS_PER_BLOCK>>>(x, s, ap_aff, ad_aff, dx_aff,\
                                                ds_aff, n, batches);
        cudaDeviceSynchronize();
        update_vars<<<batches, THREADS_PER_BLOCK>>>(x, s, y, dx_aff, ds_aff, dy_aff, ap_aff,\
                                ad_aff, m, n, batches);
        cudaDeviceSynchronize();
        // writeVec(x, n, "x_upd.txt");
        // writeVec(s, n, "s_upd.txt");
        cout << "Cost: " << endl;
        dot<<<batches, THREADS_PER_BLOCK>>>(c, x, cost, n, batches);
        cudaDeviceSynchronize();
        stop_t = second();
        time_post = stop_t - start_t;
        for (int j = 0; j < min(batches, 10); j++)
            cout << cost[j]*n << " ";
        cout << endl;
        cout << "Preops time: " << time_preop << endl;
        cout << "ADA time: " << time_ada << endl;
        cout << "Postop_ts time: " << time_post << endl;
        cout << "GEMV time: " << time_gemv << endl;
    }
    //for (int i  = 0; i < batches; i++)
      //  cout << mu[i] << endl;
    /*for (int i = 0; i < batches; i++){
        for (int j = 0; j < n; j++)
            cout << rd[i*n + j] << " ";
        cout << endl;
    }
    printVectordouble(csrValADA, 2*batches);
    printVectordouble(d_i1, m*batches);
    printVectordouble(d_i2, m*batches);
    printVectordouble(d_i3, m*batches);*/
    
    
    // printVectordouble(x, n);
    // writeVec(x, n, "x_val.txt");
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < num_streams; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken CPU: %.4f\n", ms);

    cudaFree(A);
    cudaFree(AD);
    cudaFree(AS);
    cudaFree(ADA);
    cudaFree(x);
    cudaFree(s);
    cudaFree(y);
    cudaFree(c);
    cudaFree(b);
    cudaFree(dx_aff);
    cudaFree(dy_aff);
    cudaFree(d_i1);
    cudaFree(d_i2);
    cudaFree(d_i3);
    cudaFree(ds_aff);
    cudaFree(d_i4);
    cudaFree(rd);
    cudaFree(rp);
    cudaFree(rc);
    cudaFree(v);
    cudaFree(ap_aff);
    cudaFree(ad_aff);
    cudaFree(mu_aff);
    cudaFree(mu);
    cudaFree(cost);
}

void printVectordouble(const double *V, int m)
{
    for (int i = 0; i < m; i++)
        std::cout << V[i] << " ";
    std::cout << std::endl;
}
// Generate the three vectors A, IA, JA 
// void sparesify(double *M, int m, int n, double *A, int *IA, int *JA)
// {
//     //int m = M.size(), n = M[0].size();
//     int i, j;
//     //vi A;
//     IA[0] = 1; // IA matrix has N+1 rows
//     //vi JA;
//     int NNZ = 0;
  
//     for (i = 0; i < m; i++) {
//         for (j = 0; j < n; j++) {
//             if (M[i + m*j] != 0) {
//                 A[NNZ] = M[i + m*j];
//                 JA[NNZ] = j + 1;
  
//                 // Count Number of Non Zero 
//                 // Elements in row i
//                 NNZ++;
//             }
//         }
//         IA[i + 1] = NNZ + 1;
//     }
  
//     printMatrix(M, m, n);
//     printVectordouble(A, NNZ);
//     printVector(IA, m + 1);
//     printVector(JA, NNZ);
// }
//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
 void printMatrix(const double *A, int nr_rows_A, int nr_cols_A) {
 
     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
             std::cout << A[j * nr_rows_A + i] << " ";
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }
  
// Utility Function to print A, IA, JA vectors
// with some decoration.
void printVector(const int *V, int m)
{
    for (int i = 0; i < m; i++)
        std::cout << V[i] << " ";
    std::cout << std::endl;
}

/*
* Find Number of Non Zero Elements in a Sparse Matrix
*/
int findNNZ(const double *M, int N)
{
    int nnz = 0;
    for (int i = 0;  i < N; i++)
    {
        if (M[i] != 0.0)
            nnz++;
    }

    return nnz;
}

//Get matrix dimensions of CSR Matrix
void getMatSize(string fname, int *rows, int *nnz)
{
    ifstream infile;
    infile.open(fname);
    string line;
    getline(infile, line);
    int i, n = 0;
    stringstream stream(line);
    
    while (stream >> i)
        n++;
    *rows = n - 1;
    
    getline(infile, line);
    n = 0;
    stringstream stream1(line);
    while (stream1 >> i)
        n++;
    *nnz = n;
}

// Load CSR matrix
void loadMat(double *A, int *IA, int *JA, string fname, int *rows, int *nnz)
{
    ifstream infile;
    infile.open(fname);
    string line;
    getline(infile, line);
    int i, n = 0;
    //cout << line << endl;
    stringstream stream(line);
    while (stream >> i)
    {
        IA[n] = i-1;
        n++;
    }
    *rows = n - 1;
    getline(infile, line);
    n = 0;
    stringstream stream1(line);
    while (stream1 >> i)
    {
        JA[n] = i-1;
        n++;
        //if (n < 50)
            //cout << i << " ";
    }
    cout << endl;
    *nnz = n;
    getline(infile, line);
    n = 0;
    double f;
    stringstream stream2(line);
    while (stream2 >> f)
    {
        A[n] = f;
        n++;
        //if (n < 50)
            //cout << f << " ";
    }
    cout << *rows << " " << *nnz << " " << n << endl;
}

// Load Vector
void loadVec(double *V, string fname)
{
    int cols = 0, rows = 0;
    ifstream infile;
    infile.open(fname);
    while(!infile.eof())
    {
        string line;
        getline(infile, line);
        int temp_cols = 0;
        stringstream stream(line);
        double f;
        while(stream >> f)
        {
            V[cols*rows + temp_cols++] = f;
        }
        
        if(temp_cols == 0)
            continue;
        
        if (cols == 0)
            cols = temp_cols;
        
        rows++;
    }
    cout << rows << " " << cols << endl;
    infile.close();
}

void writeVec(double *V, int n, string fname)
{
    ofstream outdata;
    outdata.open(fname);
    for (int i = 0; i < n; i++)
    {
        outdata << V[i] << endl;
    }
    outdata.close();
}
void writeVecInt(int *V, int n, string fname)
{
    ofstream outdata;
    outdata.open(fname);
    for (int i = 0; i < n; i++)
    {
        outdata << V[i] << endl;
    }
    outdata.close();
}

void createA(double *A, double *csrValA, int *csrRowA, int *csrColA, int m, int n, int nnz, int baseA)
{
    for (int i  = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i + m*j] = 0;

    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowA[i] - baseA; j < csrRowA[i+1] - baseA; j++)
        {
            A[i + m*(csrColA[j]-baseA)] = csrValA[j];
        }
    }
}






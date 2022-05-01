#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// #include "cublas_utils.h"
#include <cusolverDn.h>
#include "cusolver_utils.h"

using data_type = double;
#define THREADS_PER_BLOCK (256)
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
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
                AD[i + m * j] = A[i + m * j] * x[j];
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

                AD[i + m * j] = A[i + m * j] * x[j] / zy;
            }
        }
    }
}

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
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
void gemm_batched(int m, int k, int n, int batch_count, int lda, int ldb, int ldc, data_type *A_batch, data_type *B_batch, data_type *C_batch)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

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


    printf("A matrix\n");
    for (int i = 0; i < batch_count; i++)
        printM(m, n, &A_batch[m * k * i], lda);
    printf("-------\n");

    printf("B matrix\n");
    for (int i = 0; i < batch_count; i++)
        printM(m, n, &B_batch[k * n * i], ldb);
    printf("-------\n");
    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    
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
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("C matrix\n");
    for (int i = 0; i < batch_count; i++)
        printM(m, n, &C_batch[m * n * i], ldc);
    printf("-------\n");
    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

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
void gemv_batched(int m, int k, int batch_count, int lda, data_type *A_batch, data_type *B_batch, data_type *C_batch)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int n = 1;
    // const int lda = m;
    const int ldb = k;
    const int ldc = m;
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

    // data_type *A_batch;
    // cudaMallocManaged(A_batch, sizeof(data_type) * m * k);

    printf("A[0]\n");
    print_matrix(m, k, &A_batch[0], lda);
    printf("=====\n");

    printf("A[1]\n");
    print_matrix(m, k, &A_batch[m * k], lda);
    printf("=====\n");

    printf("B[0]\n");
    print_matrix(k, n, &B_batch[0], ldb);
    printf("=====\n");

    printf("B[1]\n");
    print_matrix(k, n, &B_batch[k * n], ldb);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

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

    // /* step 3: compute */
    // CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
    //                                 d_B_array, ldb, &beta, d_C_array, ldc, batch_count));
    /* step 3: compute */
    CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, n, m, k, &alpha, d_B_array, n,
                                    d_A_array, k, &beta, d_C_array, n, batch_count));
    /* step 4: copy data to host */
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(&C_batch[i * m * n], d_C[i], sizeof(data_type) * m * n,
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
     *       | 43.0 | 50.0 | 151.0 | 166.0 |
     */
    printf("C matrix\n");
    for (int i = 0; i < batch_count; i++)
        printM(m, n, &C_batch[m * n * i], ldc);
    printf("-------\n");
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

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

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

    std::printf("A0 = (matlab base-1)\n");
    print_matrix(m, m, &A_batch[0], lda);
    std::printf("=====\n");

    std::printf("A1 = (matlab base-1)\n");
    print_matrix(m, m, &A_batch[m * m], lda);
    std::printf("=====\n");

    std::printf("B0 = (matlab base-1)\n");
    print_matrix(m, 1, &B_batch[0], ldb);
    std::printf("=====\n");
    std::printf("B1 = (matlab base-1)\n");
    print_matrix(m, 1, &B_batch[m], ldb);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    for (int j = 0; j < batchSize; j++) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Aarray[j]), sizeof(double) * lda * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Barray[j]), sizeof(double) * ldb * nrhs));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * infoArray.size()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double *) * Aarray.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Barray), sizeof(double *) * Barray.size()));

    CUDA_CHECK(cudaMemcpyAsync(Aarray[0], &A_batch[0], sizeof(double) * m * m,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(Aarray[1], &A_batch[m*m], sizeof(double) * m * m,
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(
        cudaMemcpyAsync(Barray[0], &B_batch[0], sizeof(double) * m, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(Barray[1], &B_batch[m], sizeof(double) * m, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double) * Aarray.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Barray, Barray.data(), sizeof(double) * Barray.size(),
                               cudaMemcpyHostToDevice, stream));
    
    /* step 3: Cholesky factorization */
    CUSOLVER_CHECK(
        cusolverDnDpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_infoArray, batchSize));

    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int) * infoArray.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(L0.data(), Aarray[0], sizeof(double) * lda * m,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int j = 0; j < batchSize; j++) {
        std::printf("info[%d] = %d\n", j, infoArray[j]);
    }

    std::printf("L = (matlab base-1), upper triangle is don't care \n");
    print_matrix(m, m, L0.data(), lda);
    std::printf("=====\n");


    /*
     * step 4: solve A0*X0 = B0
     *        | 1 |        | 10.5 |
     *   B0 = | 1 |,  X0 = | -2.5 |
     *        | 1 |        | -1.5 |
     */
    CUSOLVER_CHECK(cusolverDnDpotrsBatched(cusolverH, uplo, m, nrhs, /* only support rhs = 1*/
                                           d_Aarray, lda, d_Barray, ldb, d_infoArray, batchSize));

    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    for (int i = 0; i < batchSize; i++)
    {
        CUDA_CHECK(
        cudaMemcpyAsync(&C_batch[i * m], Barray[i], sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    }
    

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after potrsBatched: infoArray[0] = %d\n", infoArray[0]);
    if (0 > infoArray[0]) {
        std::printf("%d-th parameter is wrong \n", -infoArray[0]);
        exit(1);
    }

    std::printf("X0 = (matlab base-1)\n");
    print_matrix(m, 1, &C_batch[0], ldb);
    std::printf("=====\n");

    std::printf("X1 = (matlab base-1)\n");
    print_matrix(m, 1, &C_batch[m], ldb);
    std::printf("=====\n");
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

    CUDA_CHECK(cudaDeviceReset());

}
int main(int argc, char *argv[]) {
    // cublasHandle_t cublasH = NULL;
    // cudaStream_t stream = NULL;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;
    const int batch_count = 2;

    /*
     *   A = | 1.0 | 2.0 | 5.0 | 6.0 |
     *       | 3.0 | 4.0 | 7.0 | 8.0 |
     *
     *   B = | 5.0 | 6.0 |  9.0 | 10.0 |
     *       | 7.0 | 8.0 | 11.0 | 12.0 |
     */

    // const std::vector<std::vector<data_type>> A_array = {{1.0 ,3.0, 2.0, 4.0},
    //                                                      {5.0, 7.0, 6.0, 8.0}};
    // const std::vector<std::vector<data_type>> B_array = {{5.0, 7.0, 6.0, 8.0},
    //                                                      {9.0, 11.0, 10.0, 12.0}};
    // std::vector<std::vector<data_type>> C_array(batch_count, std::vector<data_type>(m * n));

    // const data_type alpha = 1.0;
    // const data_type beta = 0.0;

    // data_type **d_A_array = nullptr;
    // data_type **d_B_array = nullptr;
    // data_type **d_C_array = nullptr;

    // std::vector<data_type *> d_A(batch_count, nullptr);
    // std::vector<data_type *> d_B(batch_count, nullptr);
    // std::vector<data_type *> d_C(batch_count, nullptr);

    // cublasOperation_t transa = CUBLAS_OP_N;
    // cublasOperation_t transb = CUBLAS_OP_N;

    // printf("A[0]\n");
    // print_matrix(m, k, A_array[0].data(), lda);
    // printf("=====\n");

    // printf("A[1]\n");
    // print_matrix(m, k, A_array[1].data(), lda);
    // printf("=====\n");

    // printf("B[0]\n");
    // print_matrix(k, n, B_array[0].data(), ldb);
    // printf("=====\n");

    // printf("B[1]\n");
    // print_matrix(k, n, B_array[1].data(), ldb);
    // printf("=====\n");



    /**Diagonal Matrix Multiplication AD = A * D **/
    double *A, *AD, *x, *y, *B, *C, *V, *X;
    cudaMallocManaged(&A, sizeof(double) * m * k * batch_count);
    cudaMallocManaged(&B, sizeof(double) * k * n * batch_count);
    cudaMallocManaged(&C, sizeof(double) * m * n * batch_count);
    cudaMallocManaged(&V, sizeof(double) * n * batch_count);
    cudaMallocManaged(&X, sizeof(double) * n * batch_count);
    cudaMallocManaged(&AD, sizeof(double) * m * n * batch_count);
    cudaMallocManaged(&x, sizeof(double) * n * batch_count);
    cudaMallocManaged(&y, sizeof(double) * n * batch_count);
    for (int i = 0; i < m * n * batch_count; i++)
    {
        A[i] = i % 4 + 1;
        if (i % 3 == 0)
            A[i] = 7;
        B[i] = 2;
    }
       
    for (int i = 0; i < n * batch_count; i++)
    {
        x[i] = (i) % batch_count + 1;
        y[i] = (i) % batch_count + 1;
        X[i] = (i) % batch_count + 1;
    }

    printf("A matrix\n");
    for (int i = 0; i < batch_count; i++)
        print_matrix(m, n, &A[m * n * i], lda);
    printf("-------\n");

    printf("Vetors\n");
    // print_vector(n, x);
    // print_vector(n, y);

    printf("AD matrix 1\n");
    diag_matmul<<<batch_count, THREADS_PER_BLOCK>>>(m, n, batch_count, 0, A, x, y, AD, 0);
    cudaDeviceSynchronize();
    for (int i = 0; i < batch_count; i++)
        print_matrix(m, n, &AD[m * n * i], lda);
    printf("-------\n");

    printf("AD matrix 2\n");
    diag_matmul<<<batch_count, THREADS_PER_BLOCK>>>(m, n, batch_count, 1, A, x, y, AD, 0);
    cudaDeviceSynchronize();
    for (int i = 0; i < batch_count; i++)
        print_matrix(m, n, &AD[m * n * i], lda);
    printf("-------\n");


    // gemm_batched(m, k, n, batch_count, lda, ldb, ldc, AD, A, C);
    gemv_batched(m, k, batch_count, lda, A, X, V);
    //ports_batched(m, batch_count, C, X, V);


    // /* step 1: create cublas handle, bind a stream */
    // CUBLAS_CHECK(cublasCreate(&cublasH));

    // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // /* step 2: copy data to device */
    // for (int i = 0; i < batch_count; i++) {
    //     CUDA_CHECK(
    //         cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(data_type) * A_array[i].size()));
    //     CUDA_CHECK(
    //         cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(data_type) * B_array[i].size()));
    //     CUDA_CHECK(
    //         cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * C_array[i].size()));
    // }

    // CUDA_CHECK(
    //     cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * batch_count));
    // CUDA_CHECK(
    //     cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * batch_count));
    // CUDA_CHECK(
    //     cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * batch_count));

    // for (int i = 0; i < batch_count; i++) {
    //     CUDA_CHECK(cudaMemcpyAsync(d_A[i], A_array[i].data(), sizeof(data_type) * A_array[i].size(),
    //                                cudaMemcpyHostToDevice, stream));
    //     CUDA_CHECK(cudaMemcpyAsync(d_B[i], B_array[i].data(), sizeof(data_type) * B_array[i].size(),
    //                                cudaMemcpyHostToDevice, stream));
    // }

    // CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type *) * batch_count,
    //                            cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * batch_count,
    //                            cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * batch_count,
    //                            cudaMemcpyHostToDevice, stream));

    // /* step 3: compute */
    // CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
    //                                 d_B_array, ldb, &beta, d_C_array, ldc, batch_count));

    // /* step 4: copy data to host */
    // for (int i = 0; i < batch_count; i++) {
    //     CUDA_CHECK(cudaMemcpyAsync(C_array[i].data(), d_C[i], sizeof(data_type) * C_array[i].size(),
    //                                cudaMemcpyDeviceToHost, stream));
    // }

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // /*
    //  *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
    //  *       | 43.0 | 50.0 | 151.0 | 166.0 |
    //  */

    // printf("C[0]\n");
    // print_matrix(m, n, C_array[0].data(), ldc);
    // printf("=====\n");

    // printf("C[1]\n");
    // print_matrix(m, n, C_array[1].data(), ldc);
    // printf("=====\n");

    // /* free resources */
    // CUDA_CHECK(cudaFree(d_A_array));
    // CUDA_CHECK(cudaFree(d_B_array));
    // CUDA_CHECK(cudaFree(d_C_array));
    // for (int i = 0; i < batch_count; i++) {
    //     CUDA_CHECK(cudaFree(d_A[i]));
    //     CUDA_CHECK(cudaFree(d_B[i]));
    //     CUDA_CHECK(cudaFree(d_C[i]));
    // }

    // CUBLAS_CHECK(cublasDestroy(cublasH));

    // CUDA_CHECK(cudaStreamDestroy(stream));

    // CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
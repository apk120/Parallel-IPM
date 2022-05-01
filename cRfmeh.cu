#include <iostream>
#include <sstream>
#include <assert.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#include <cusolverRf.h>
#include <cuda_runtime.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include "inc/helper_string.h"
#include "inc/helper_cusolver.h"
#include "inc/helper_cuda.h"
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

using namespace std;
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

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
#define THREADS_PER_BLOCK 256

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
*
*/
__global__ void findmuAff(double *x_b, double *s_b, double *mu, double *ap_b, double *dx_b, double *ad_b, double *ds_b, int n, int no_batch)
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
    double *dx = &dx_b[batch * n];
    double *ds = &ds_b[batch * n];
    double ap = ap_b[batch];
    double ad = ad_b[batch];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        temp[i%THREADS_PER_BLOCK] += (x[i] + ap * dx[i])* (s[i] + ad * ds[i]);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        mu[batch] = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++)
            mu[batch] += temp[i]/n;
    }
}
/*
* Multiplication of a CSR Sparse Matrix with a Vector
*/
__global__ void csr_mul_Av(double *csrValA, int *csrRowA,\
             int *csrColA, double *B_b, double *res_b, int m, int n, int nnz, int no_batch, int baseA)
{  
    int batch;
    //__shared__ double sum[THREADS_PER_BLOCK];
    __shared__ int i[THREADS_PER_BLOCK];
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else 
        return;

    double *B = &B_b[batch * n];
    double *res = &res_b[batch * m];

    for (int j = threadIdx.x; j < m; j += blockDim.x)
    {
        double sum = 0.0;
        for (i[threadIdx.x] = csrRowA[j] - baseA; i[threadIdx.x] < csrRowA[j + 1] - baseA; i[threadIdx.x]++)
        {
            if (i[threadIdx.x] >= nnz)
                printf("wrrong, %d %d %d m = %d, n = %d\n", i[threadIdx.x], j, threadIdx.x, m, n);
            sum += csrValA[i[threadIdx.x]] * B[csrColA[i[threadIdx.x]] - baseA]; //B[]
        }     
        res[j] = sum;
    }  
}

/*
* Multiplication of CSR Matrix with a Diagonal Matrix (Batched)
* A is fixed, d (Diagonal Elements of Diagonal Matrix) varies
* if (div == 0) A * d
* else A * d / y
*/
__global__ void csr_diag_matmul(double *csrValA, int *csrRowA, int *csrColA, double *csrValres_b,\
                    double *d_b, double *y_b, int div, int m, int n, int nnz, int no_batch, int baseA)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;

    double *d = &d_b[batch * n];
    double *y = &y_b[batch * n];
    double *csrValRes = &csrValres_b[batch * nnz];
    double zy = 1.0;
    for (int j = threadIdx.x; j < nnz; j += blockDim.x)
    {
        if (div == 0)
            csrValRes[j] = csrValA[j] * d[csrColA[j] - baseA];
        else if (div == 1)
        {
            double sign = 1.0;
            if (y[csrColA[j] - baseA] < 0)
                sign = -1.0;
            
            if (abs(y[csrColA[j] - baseA]) < 1e-9)
                zy = 1e-9*sign;
            else
                zy = y[csrColA[j] - baseA];
            csrValRes[j] = csrValA[j] / zy;
        }
        else
        {
            double sign = 1.0;
            if (y[csrColA[j] - baseA] < 0)
                sign = -1.0;
            
            if (abs(y[csrColA[j] - baseA]) < 1e-9)
                zy = 1e-9*sign;
            else
                zy = y[csrColA[j] - baseA];
            csrValRes[j] = csrValA[j] * (d[csrColA[j] - baseA]) / zy;//+1e-13
        }   
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
*
*/
__global__ void ADAt(double *csrValA, int *csrRowA, int *csrColA,\
            double *csrValADA_b, int *csrRowADA_b, int *csrColADA_b, double *x_b, \
            double *y_b, int m, int n, int nnz, int nnzAAt, int no_batch)
{
    int batch;
    __shared__ double sum[THREADS_PER_BLOCK];
    __shared__ int k[THREADS_PER_BLOCK];
    __shared__ int j[THREADS_PER_BLOCK];
    __shared__ int a[THREADS_PER_BLOCK];
    __shared__ int b[THREADS_PER_BLOCK];
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    double *x = &x_b[batch * n];
    double *y = &y_b[batch * n];
    int *csrRowADA = &csrRowADA_b[batch * (m+1)];
    int *csrColADA = &csrColADA_b[batch * nnzAAt];
    double *csrValADA = &csrValADA_b[batch * nnzAAt];

    for (int i = threadIdx.x; i < m; i += blockDim.x)
    {
        for (j[threadIdx.x] = csrRowADA[i] - 1; j[threadIdx.x] < csrRowADA[i+1] - 1; j[threadIdx.x]++)
        {
            k[threadIdx.x] = csrColADA[j[threadIdx.x]] - 1;
            sum[threadIdx.x] = 0.0;
            for (a[threadIdx.x] = csrRowA[i] - 1; a[threadIdx.x] < csrRowA[i + 1] - 1; a[threadIdx.x]++)
            {
                for (b[threadIdx.x] = csrRowA[k[threadIdx.x]] - 1; b[threadIdx.x] < csrRowA[k[threadIdx.x]+1]-1; b[threadIdx.x]++)
                {
                    if (csrColA[a[threadIdx.x]] == csrColA[b[threadIdx.x]])
                    {
                        double zy = y[csrColA[a[threadIdx.x]]-1];
                        double sign = 1.0;
                        if (zy < 0)
                            sign = -1.0;
                        if (abs(zy) < 1e-9)
                            zy = 1e-9 *sign;
                        sum[threadIdx.x] += csrValA[a[threadIdx.x]] * csrValA[b[threadIdx.x]] * x[csrColA[a[threadIdx.x]]-1]/zy;
                    }
                }
            }
            csrValADA[j[threadIdx.x]] = sum[threadIdx.x];
        }
    }
}

__global__ void ADAT(double *A, double *csrValAD_b, int *csrRowAD, int *csrColAD,\
            double *csrValADA_b, int *csrRowADA, int *csrColADA, double *x_b, \
            double *y_b, int m, int n, int nnz, int nnzAAt, int no_batch, int baseA)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    //double *x = &x_b[batch * n];
    //double *y = &y_b[batch * n];
    double *csrValADA = &csrValADA_b[batch * nnzAAt];
    
    double *csrValAD = &csrValAD_b[batch * nnz];

    for (int i = threadIdx.x; i < m; i += blockDim.x)
    {
        for (int j = csrRowADA[i] - baseA; j < csrRowADA[i+1] - baseA; j++)
        {
            if (j < 0)
                printf("Error %d\n", j);
            int k = csrColADA[j] - baseA;
            double sum = 0.0;
            for (int a = csrRowAD[i]-baseA; a < csrRowAD[i+1]-baseA; a++)
            {
                // if (k + (csrColAD[a]-baseA)*m > no_batch*m*n)
                //     printf("Wrong %d batch %d a %d csrrCol %d m %d k %d i %d\n", k + (csrColAD[a]-baseA)*m, batch, a, csrColAD[a], m, k, i);
                if (csrColAD[a] < 0)
                    printf("Worng nnz %d\n", a);
                sum += csrValAD[a] * A[k + (csrColAD[a]-baseA)*m];
            }
            if (j >= nnzAAt)
                printf("Worng nnzAAt %d", j);
            csrValADA[j] = sum + 1e-8;
        }
    }
}

void AAtransposeCPU(double *csrValA, int *csrRowA, int *csrColA,\
            double *csrValADA, int *csrRowADA, int *csrColADA, int m, int n, int nnz)
{
    int nnzRes = 0;
    for (int j = 0; j < m; j++)
    {
        for (int k = 0; k < m; k++)
        {
            double sum = 0.0;
            for (int i = csrRowA[j] - 1; i < csrRowA[j+1] - 1; i++)
            {
                for (int m = csrRowA[k] - 1; m < csrRowA[k+1] - 1; m++)
                {
                    if (csrColA[i] == csrColA[m]) 
                    {
                        sum += csrValA[i] * csrValA[m];
                    }
                }
            }
            if (sum != 0.0)
            {
                csrValADA[nnzRes] = sum;
                csrColADA[nnzRes] = k + 1;
                nnzRes++;
            }
        }
        csrRowADA[j + 1] = nnzRes + 1;
    }
    
    csrRowADA[0] = 1;
    //for (int i = 1; i <= m; i++)
      //  csrRowADA[i] = csrRowADA[i-1] + csrRowADA[i];
}


int main(int argc, char *argv[])
{
    int m = 6590, n = 12879, batches = 8, iter = 50, nnz; // 6590, 12879, 2036312
    double *csrValA,  *csrValADA, *A;//*A
    int *csrColA, *csrRowA, *csrColADA, *csrRowADA, *csrColAAT, *csrRowAAT;
    int *csrColAT, *csrRowAT;// *csrColAS, *csrRowAS;
    double *csrValAT, *csrValAS, *csrValAD, *csrValAAT;
    double *x, *y, *s, *c, *b, *dx_aff, *dy_aff, *ds_aff, *rd, *rp, *rc, *v, *ds_cor, *dy_cor;   
    double *d_i1, *d_i2, *d_i3, *d_i4;
    //double sigma = 0.8;
    double *ap_aff, *ad_aff, *mu_aff, *mu, *cost, *sigma;
    int nnz_aat = 2042300;
    int base = 0;
    struct testOpts opts;
    double start_t, stop_t, time_preop, time_ada, time_post;
    double min_norm;
    int rankAt;
    int *perm;

    parseCommandLineArguments(argc, argv, opts);
    findCudaDevice(argc, (const char **)argv);

    string fname = "A_sparse.csv";
    getMatSize(fname, &m, &nnz);
    printf("Non Zero Elements: %d, rows: %d\n", nnz, m);

    cudaMallocManaged(&A, sizeof(double)*m*n);
    CUDIE0();
    cudaMallocManaged(&csrValA, sizeof(double)*nnz);
    cudaMallocManaged(&csrRowA, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColA, sizeof(int)*nnz);
    cudaMallocManaged(&csrValAT, sizeof(double)*nnz);
    cudaMallocManaged(&csrRowAT, sizeof(int)*(n+1));
    cudaMallocManaged(&csrColAT, sizeof(int)*nnz);
    cudaMallocManaged(&csrValAS, sizeof(double)*nnz*batches);
    cudaMallocManaged(&csrValAD, sizeof(double)*nnz*batches);
    cudaMallocManaged(&csrValADA, sizeof(double)*nnz_aat*batches);
    cudaMallocManaged(&csrRowADA, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColADA, sizeof(int)*nnz_aat);
    cudaMallocManaged(&csrValAAT, sizeof(double)*nnz_aat);
    cudaMallocManaged(&csrRowAAT, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColAAT, sizeof(int)*nnz_aat);
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
    cudaMallocManaged(&mu_aff, sizeof(double) * batches);
    cudaMallocManaged(&sigma, sizeof(double) * batches);
    cudaMallocManaged(&ds_cor, sizeof(double) * n *batches);
    cudaMallocManaged(&dy_cor, sizeof(double) * m * batches);
    cudaMallocManaged(&perm, sizeof(int) * m * batches);
    // cudaMallocManaged(&v, sizeof(double)* n * batches);
    cudaMallocManaged(&cost, sizeof(double) * batches);
    CUDIE0();
    /*Initialization of the Problem*/
    double *ib, *ic, *ix, *is, *iy;
    ib = (double *)malloc(m * sizeof(double));
    ic = (double *)malloc(n * sizeof(double));
    ix = (double *)malloc(n * sizeof(double));
    is = (double *)malloc(n * sizeof(double));
    iy = (double *)malloc(m * sizeof(double));
    loadVec(ib, "Btxt.csv");
    loadVec(ic, "Ctxt.csv");
    loadVec(ix, "x_initial.csv");
    loadVec(is, "s_initial.csv");
    loadVec(iy, "y_initial.csv");
    //double ib[2] = {1, 2}, ic[4] = {-1, -1, 1, 1};
    //double ix[4] = {0.8, 0.1, 0.2, 1.9}, is[4] = {0.1, 0.2, 2.1, 2.2}, iy[2] = {-1.1, -1.2};
    
    double miny = 100, maxy = -100;
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < m; j++) {
            b[m*i + j] = ib[j];
            y[m*i + j] = iy[j];
            if (i == 0)
            {
                miny = min(miny, iy[j]);
                maxy = max(maxy, iy[j]);
            }
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

    loadMat(csrValA, csrRowA, csrColA, "A_sparse.csv", &m, &nnz);
    loadMat(csrValAT, csrRowAT, csrColAT, "At_sparse.csv", &n, &nnz);
    /*printVector(csrColAT, 10);
    printVectordouble(csrValAT, 10);
    printVector(csrRowAT, 10);
    writeVecInt(csrColAT, nnz, "csrColAt_bef.txt");
    writeVecInt(csrRowAT, n+1, "csrRowAt_first.txt");*/
    //writeVec(csrValAT, 1000, "csrValAt_bef.txt");
    //sparesify(A, n, m, csrValAT, csrRowAT, csrColAT);
    int nnzAAt = nnz_aat, r_aat;
    cout << "Transpose Start" << endl;
    loadMat(csrValADA, csrRowADA, csrColADA, "AAt_sparse.csv", &r_aat, &nnzAAt);
    for (int i = 1; i < batches; i++)
    {
        //AAtransposeCPU(csrValA, csrRowA, csrColA, &csrValADA[nnzAAt*i],\
        //            &csrRowADA[(m+1)*i], &csrColADA[nnzAAt*i], m, n, nnz);
        for (int j = 0; j < nnzAAt; j++)
        {
            csrValADA[nnzAAt*i + j] = csrValADA[j];
        }
    }   
    cout << "Transpose End" << endl;
    CHECK_CUDA( cudaMemcpy(csrValAAT, csrValADA,
                           nnz_aat * sizeof(double), cudaMemcpyDefault) )
    CHECK_CUDA( cudaMemcpy(csrColAAT, csrColADA,
                           nnz_aat * sizeof(int), cudaMemcpyDefault) )
    CHECK_CUDA( cudaMemcpy(csrRowAAT, csrRowADA,
                           (m + 1) * sizeof(int), cudaMemcpyDefault) )
    /*cout << csrRowADA[1] << " " << csrRowADA[m] << endl;
    cout << csrValADA[1] <<  " " << csrValADA[nnzAAt] << endl;
    cout << csrColADA[1] <<  " " << csrColADA[nnzAAt+1] << endl; 
    cout << "Transpose End" << endl;*/
    CUDIE0();
    createA(A, csrValA, csrRowA, csrColA, m, n, nnz, base);
    /*printVectordouble(csrValADA, nnzAAt * batches);
    printVector(csrRowADA, (m + 1)*batches);
    printVector(csrColADA, nnzAAt * batches);*/


    // Handle creation
    cusolverSpHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverSpCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverSpSetStream(cusolverH, stream);

    //mat descriptor for AT
    cusparseMatDescr_t descrAt = NULL;
    
    checkCudaErrors(cusparseCreateMatDescr(&descrAt));
    checkCudaErrors(cusparseSetMatType(descrAt, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatDiagType(descrAt, CUSPARSE_DIAG_TYPE_NON_UNIT));
    checkCudaErrors(cusparseSetMatIndexBase(descrAt, CUSPARSE_INDEX_BASE_ZERO));
    
    /*---Initialization*/

    //printVectordouble(x, n*batches);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;
    cudaEventRecord(start);

    /*Main Loop for IPM iterations*/
    for (int i = 0; i < iter; i++)
    {
        start_t = second();
        start_t = second();
        dot<<<batches, THREADS_PER_BLOCK>>>(x, s, mu, n, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        cout << "MU" << endl;
        for (int j = 0; j < min(batches, 10); j++)
            cout << mu[j]*n << " " << n << " ";
        cout << endl;
        //writeVecInt(csrRowAT, n+1, "csrRowAt_bef.txt");
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, y, rd,\
                                                    n, m, nnz, batches, base);
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
        
        vector_add<<<batches, THREADS_PER_BLOCK>>>(c, s, rd, rd, 1, -1, -1, n, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        // writeVec(rd, n, "yuss2.txt");
        
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA, x, rp,\
                                                    m, n, nnz, batches, base);
        cudaDeviceSynchronize();
        CUDIE0();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(b, rp, rp, rp, 1, -1, 0, m, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        //writeVec(rp, m, "yuss1.txt");
        vector_mul<<<batches, THREADS_PER_BLOCK>>>(x, s, mu, rc, 0, -1, n, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        
        csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
                                 csrValAS, s, s, 1, m, n, nnz, batches, base);
        cudaDeviceSynchronize();
        CUDIE0();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAS, csrRowA, csrColA, rc, d_i1,\
                                                    m, n, nnz, batches, base);
        cudaDeviceSynchronize();
        CUDIE0();
        csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
                                 csrValAD, x, s, 2, m, n, nnz, batches, base);
        cudaDeviceSynchronize();
        CUDIE0();

        stop_t = second();
        time_preop = stop_t - start_t;

        cout << "ADAT start_t\n";
        start_t = second();
        start_t = second();
        
        ADAT<<<batches, THREADS_PER_BLOCK>>>(A, csrValAD, csrRowA, csrColA, csrValADA,\
                               csrRowADA, csrColADA, x, s, m, n, nnz, nnzAAt, batches, base);
        int crr = 0;
        cout << "ADAT start_t\n";
        // int ccmn(int *hA_csrOffsets, int *hA_columns, double *hA_values,int *hB_csrOffsets, int *hB_columns, double *hB_values, int *hC_csrOffsets, int *hC_columns, double *hC_values, int C_nnz, double *resC) 
        // crr = xsc(csrRowA, csrColA, csrValAD, csrRowAT, csrColAT, csrValAT, csrRowADA, csrColADA, csrValAAT, nnzAAt, csrValADA);
        //return 0;
        if (crr != 0)
        {
            printf("Error\n");
            return 0;
        }
        cudaDeviceSynchronize();
        CUDIE0();
        stop_t = second();
        time_ada = stop_t - start_t;
        /*writeVec(csrValADA, nnzAAt, "csrValADA.txt");
        writeVecInt(csrColADA, nnzAAt, "csrColADA.txt");
        writeVecInt(csrRowADA, m + 1, "csrRowADA.txt");*/
        cout << "ADAT End\n";
        //return 0;
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAD, csrRowA, csrColA, rd, d_i2,\
                                                    m, n, nnz, batches, base);
        cudaDeviceSynchronize();
        CUDIE0();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(d_i1, d_i2, rp, d_i3, -1, 1, 1, m, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        
        //writeVec(d_i2, m, "yuss3.txt");
        //writeVec(d_i3, m, "yuss.txt");
        cout << m << " " << n << " " << nnzAAt<< " " << endl;
        //Solve for dy_aff
        solve(m, m, m, nnzAAt, batches, base, csrValADA, csrRowADA, csrColADA, dy_aff, d_i3, opts);

        start_t = second();
        start_t = second();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, dy_aff, ds_aff,\
                                                    n, m, nnz, batches, base);
        cudaDeviceSynchronize();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(ds_aff, rd, ds_aff, ds_aff, -1, 1, 0, n, batches);
        cudaDeviceSynchronize();
        diag_vector_mul<<<batches, THREADS_PER_BLOCK>>>(s, s, rc, d_i4, n, 1, batches);
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

        if (i > 35) {
            //Change from here
            findmuAff<<<batches, THREADS_PER_BLOCK>>>(x, s, mu_aff, ap_aff, dx_aff, ad_aff, ds_aff, n, batches);
            cudaDeviceSynchronize();
            for (int j = 0; j < batches; j++)
            {
                sigma[j] = pow((mu_aff[j]/mu[j]), 3);
            }
            for (int j = 0; j < batches; j++)
            {
                mu_aff[j] = sigma[j] * mu[j];
            }
            vector_mul<<<batches, THREADS_PER_BLOCK>>>(dx_aff, ds_aff, mu_aff, v, 1, -1, n, batches);
            cudaDeviceSynchronize();
            diag_vector_mul<<<batches, THREADS_PER_BLOCK>>>(x, x, v, ds_cor, n, 1, batches);
            cudaDeviceSynchronize();
            cout << i << " LS Start\n";
            cusolverSpDcsrlsqvqrHost(cusolverH, n, m, nnz, descrAt, csrValAT, csrRowAT, csrColAT, ds_cor, 1e-10, &rankAt, dy_cor, perm, &min_norm);
            cudaDeviceSynchronize();
            cout << i << " LS End\n";
            // for (int j = 0; j < m; j++)
            // {
            //     dy_cor[j] = -1.0 * dy_cor[j];
            // }
            vector_add<<<batches, THREADS_PER_BLOCK>>>(dy_aff, dy_cor, dy_cor, dy_aff, 1, -1, 0, m, batches);
            vector_add<<<batches, THREADS_PER_BLOCK>>>(ds_aff, ds_cor, ds_cor, ds_aff, 1, 1, 0, n, batches);
            cudaDeviceSynchronize();
            //find corrector part
            find_update_param<<<batches, THREADS_PER_BLOCK>>>(x, s, ap_aff, ad_aff, dx_aff,\
                                                    ds_aff, n, batches);
            cudaDeviceSynchronize();
        }
        update_vars<<<batches, THREADS_PER_BLOCK>>>(x, s, y, dx_aff, ds_aff, dy_aff, ap_aff,\
                                ad_aff, m, n, batches);
        cudaDeviceSynchronize();








        //Till here
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
    printVectordouble(d_i3, m*batches);
    printVectordouble(x, n * batches);*/
    writeVec(x, n, "x_val.txt");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken CPU: %.4f\n", ms);

    cudaFree(A);
    cudaFree(csrValA);
    cudaFree(csrRowA);
    cudaFree(csrColA);
    cudaFree(csrValAT);
    cudaFree(csrRowAT);
    cudaFree(csrColAT);
    cudaFree(csrValAS);
    cudaFree(csrValAD);
    cudaFree(csrValADA);
    cudaFree(csrRowADA);
    cudaFree(csrColADA);
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
#define   A_NUM_ROWS 6590   // C compatibility
#define   C_NUM_NNZ 2042300   // C compatibility
int ccmn(int *hA_csrOffsets, int *hA_columns, double *hA_values,int *hB_csrOffsets, int *hB_columns, double *hB_values, int *hC_csrOffsets, int *hC_columns, double *hC_values, int C_nnz, double *resC) 
{
    printf("In\n"); fflush(stdout);
    //--------------------------------------------------------------------------
    //Device memory management: Allocate and copy A, B
    int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
          *dC_csrOffsets, *dC_columns;
    double *dA_values, *dB_values, *dC_values;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_64F;
    double alpha = 1.0, beta = 0.0;
    int A_num_rows = 6590;
    int A_num_cols = 12879;
    int A_nnz = 483020;
    int batches = 16;
    int B_num_rows = A_num_cols;
    int B_num_cols = A_num_rows;
    int B_nnz = A_nnz;
    // allocate A
    printf("0\n");
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double)) )
    // allocate B
    CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets,
                           (B_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_columns, B_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_values,  B_nnz * sizeof(double)) )
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    // copy A
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_nnz * sizeof(double), cudaMemcpyHostToDevice) )
    // copy B
    CHECK_CUDA( cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                           (B_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                           B_nnz * sizeof(double), cudaMemcpyHostToDevice) )

    // //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL;
    void*  dBuffer2    = NULL;
    void*  dBuffer3    = NULL;
    void*  dBuffer4    = NULL;
    void*  dBuffer5    = NULL;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;
    size_t bufferSize4 = 0;
    size_t bufferSize5 = 0;
    printf("1\n");
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    //==========================================================================
    // SpGEMM Computation
    //==========================================================================
    printf("2\n");
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize1, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize1, dBuffer1)
    )
    // //--------------------------------------------------------------------------

    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                &bufferSize2, NULL, &bufferSize3, NULL,
                                &bufferSize4, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer3, bufferSize3) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer4, bufferSize4) )
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
                                &bufferSize4, dBuffer4)
    )
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    //--------------------------------------------------------------------------
    printf("3\n");
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(double)) )
    CHECK_CUDA( cudaMemset(dC_values, 0x0, C_nnz1 * sizeof(double)) )
    // fill dC_values if needed
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )
    //--------------------------------------------------------------------------

    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                 CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                 &bufferSize5, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer5, bufferSize5) )
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                 CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                 &bufferSize5, dBuffer5)
    )
    CHECK_CUDA( cudaFree(dBuffer3) )
    printf("4\n");
    // //--------------------------------------------------------------------------
  
    printf("5\n");
    for (int i = 0; i < batches; i++)
    {
        // update dA_values, dB_values
        CHECK_CUDA( cudaMemcpy(dA_values, &hA_values[i * A_nnz],
                            A_nnz * sizeof(double), cudaMemcpyDefault) )

        CHECK_CUDA( cudaMemcpy(dB_values, hB_values,
                           B_nnz * sizeof(double), cudaMemcpyHostToDevice) )
        CHECK_CUSPARSE(
        cusparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
                                    matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc)
        )
        

        CHECK_CUDA( cudaMemcpy(&resC[i * C_nnz], dC_values,
                            C_nnz * sizeof(double), cudaMemcpyDefault) )
    }
    printf("6\n");
    //==========================================================================
    //==========================================================================
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // //--------------------------------------------------------------------------
    // // device result check
    // int   hC_csrOffsets_tmp[A_NUM_ROWS + 1];
    // int   hC_columns_tmp[C_NUM_NNZ];
    // double hC_values_tmp[C_NUM_NNZ];
    // CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
    //                        (A_num_rows + 1) * sizeof(int),
    //                        cudaMemcpyDeviceToHost) )
    // CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz * sizeof(int),
    //                        cudaMemcpyDeviceToHost) )
    // CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, C_nnz * sizeof(double),
    //                        cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // printf("7\n");
    // for (int i = 0; i < A_num_rows + 1; i++) {
    //     if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i]) {
    //         correct = 0;
    //         break;
    //     }
    // }
    // for (int i = 0; i < C_nnz; i++) {
    //     if (hC_columns_tmp[i] != hC_columns[i]) { // direct floating point
    //         correct = 0;                          // comparison is not reliable
    //         break;
    //     }
    // }
    // for (int i = 0; i < C_nnz; i++) {
    //     if (hC_columns_tmp[i] != hC_columns[i]) //||
    //         //hC_values_tmp[i]  != hC_values[i]) { // direct floating point
    //     {
    //         correct = 0;                         // comparison is not reliable
    //         break;
    //     }
    // }
    
    // if (correct)
    //     printf("spgemm_reuse_example test PASSED\n");
    // else {
    //     printf("spgemm_reuse_example test FAILED: wrong result\n");
    //     return EXIT_FAILURE;
    // }
    // //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer4) )
    CHECK_CUDA( cudaFree(dBuffer5) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrOffsets) )
    CHECK_CUDA( cudaFree(dB_columns) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return 0;
}

int xsc(int *hA_csrOffsets, int *hA_columns, double *hA_values,int *hB_csrOffsets, int *hB_columns, double *hB_values, int *hC_csrOffsets, int *hC_columns, double *hC_values, int C_nnz, double *resC) 
{
    printf("HERE\n");
    return ccmn(hA_csrOffsets,hA_columns,hA_values,hB_csrOffsets, hB_columns, hB_values,hC_csrOffsets, hC_columns, hC_values, C_nnz, resC);
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


/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  A framework of refactorization process.
 *  
 *  step 1: compute P*A*Q = L*U by
 *    - reordering and
 *    - LU with partial pivoting in cusolverSp
 *
 *  step 2: set up cusolverRf by (P, Q, L, U)
 *
 *  step 3: analyze and refactor A
 *
 *  How to use
 *     ./cuSolverRf -P=symrcm -file <file>
 *     ./cuSolverRf -P=symamd -file <file>
 *
 */



// template <typename T_ELEM>
// int loadMMSparseMatrix(
//     char *filename,
//     char elem_type,
//     bool csrFormat,
//     int *m,
//     int *n,
//     int *nnz,
//     T_ELEM **aVal,
//     int **aRowInd,
//     int **aColInd,
//     int extendSymMatrix);

void UsageRF(void)
{
    printf( "<options>\n");
    printf( "-h          : display this help\n");
    printf( "-P=<name>    : choose a reordering\n");
    printf( "              symrcm (Reverse Cuthill-McKee)\n");
    printf( "              symamd (Approximate Minimum Degree)\n");
    printf( "-file=<filename> : filename containing a matrix in MM format\n");
    printf( "-bs=<batch_size> : normally 32 - 128, default=32 \n");
    printf( "-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit( 0 );
}
/* compute | b - A*x|_inf */
void residaul_eval(
    int n,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    const double *x,
    double *r_nrminf_ptr)
{
    const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE)? 0:1 ;
    const int lower = (CUSPARSE_FILL_MODE_LOWER == cusparseGetMatFillMode(descrA))? 1:0;
    const int unit  = (CUSPARSE_DIAG_TYPE_UNIT == cusparseGetMatDiagType(descrA))? 1:0;

    double r_nrminf = 0;
    for(int row = 0 ; row < n ; row++){
        const int start = csrRowPtr[row]   - base;
        const int end   = csrRowPtr[row+1] - base;
        double dot = 0;
        for(int colidx = start ; colidx < end; colidx++){
            const int col = csrColInd[colidx] - base;
            double Aij = csrVal[colidx];
            double xj  = x[col];
            if ( (row == col) && unit ){
                Aij = 1.0;
            }
            int valid = (row >= col) && lower ||
                        (row <= col) && !lower ;
            if ( valid ){
                dot += Aij*xj;
            }
        }
        double ri = b[row] - dot;
        r_nrminf = (r_nrminf > fabs(ri))? r_nrminf : fabs(ri);
    }
    *r_nrminf_ptr = r_nrminf;
}
void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
    {
        UsageRF();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "P"))
    {
        char *reorderType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

        if (reorderType)
        {
            if ((STRCASECMP(reorderType, "symrcm") != 0) && (STRCASECMP(reorderType, "symamd") != 0))
            {
                printf("\nIncorrect argument passed to -P option\n");
                UsageRF();
            }
            else
            {
                opts.reorder = reorderType;
            }
        }
    }

    if (!opts.reorder)
    {
        opts.reorder = "symrcm"; // Setting default reordering to be symrcm.
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageRF();
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "bs"))
    {
        char *batch_size = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "bs", &batch_size);

        if (batch_size)
        {
            opts.batch_size = atoi(batch_size);
        }
        else
        {
            printf("\nIncorrect batch size passed to -bs \n ");
            UsageRF();
        }
    }
}

int solve (int rowsA, int colsA, int N, int nnzA, int batchSize, int baseA, \
            double *h_csrValA, int *h_csrRowPtrA, int *h_csrColIndA, \
             double *x, double *b, struct testOpts opts)
{
    //struct testOpts opts;
    cusolverRfHandle_t cusolverRfH = NULL; // refactorization
    cusolverSpHandle_t cusolverSpH = NULL; // reordering, permutation and 1st LU factorization
    cusparseHandle_t   cusparseH = NULL;   // residual evaluation
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL; // A is a base-0 general matrix

    csrluInfoHost_t info = NULL; // opaque info structure for LU with parital pivoting

    /*int batchSize = 32;
    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int N     = 0; // n = rowsA = colsA
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format
                   // cusolverRf only works for base-0*/

// cusolverRf only works for square matrix, 
// assume n = rowsA = colsA

    // CSR(A) from I/O
    //int *h_csrRowPtrA = NULL; // <int> n+1 
    //int *h_csrColIndA = NULL; // <int> nnzA 
    //double *h_csrValA = NULL; // <double> nnzA 
    //array of pointers to the values of each matrix in the batch (of size
    //batchSize) on the host
    double **h_A_array = NULL;
    //For example, if h_A_batch is the array (of size batchSize*nnzA) containing 
    //the values of each matrix in the batch written contiguosly one matrix  
    //after another on the host, then h_A_array[j] = &h_A_batch[nnzA*j];
    //for j=0,...,batchSize-1.
    double *h_A_batch=NULL; 

    int *h_Qreorder = NULL; // <int> n
                            // reorder to reduce zero fill-in
                            // Qreorder = symrcm(A) or Qreroder = symamd(A)
    // B = Q*A*Q^T
    int *h_csrRowPtrB = NULL; // <int> n+1
    int *h_csrColIndB = NULL; // <int> nnzA
    double *h_csrValB = NULL; // <double> nnzA
    int *h_mapBfromA = NULL;  // <int> nnzA

    double *h_x = NULL; // <double> n,  x = A \ b
    double *h_b = NULL; // <double> n, b = ones(m,1)
    double *h_r = NULL; // <double> n, r = b - A*x
    //array (of size batchSize*n*nrhs) containing the values of each rhs in 
    //the batch written contiguously one rhs after another on the host
    //nrhs is # of rhs for each system (currently only =1 is supported) 
    double *h_X_batch = NULL;
    double **h_X_array = NULL;

    // solve B*(Qx) = Q*b
    double *h_xhat = NULL; // <double> n, Q*x_hat = x
    double *h_bhat = NULL; // <double> n, b_hat = Q*b 

    size_t size_perm = 0;
    size_t size_internal = 0; 
    size_t size_lu  = 0; // size of working space for csrlu
    void *buffer_cpu = NULL; // working space for
                             // - permutation: B = Q*A*Q^T
                             // - LU with partial pivoting in cusolverSp

    // cusolverSp computes LU with partial pivoting
    //     Plu*B*Qlu^T = L*U
    //   where B = Q*A*Q^T
    //
    // nnzL and nnzU are not known until factorization is done.
    // However upper bound of L+U is known after symbolic analysis of LU.
    int *h_Plu = NULL; // <int> n
    int *h_Qlu = NULL; // <int> n

    int nnzL = 0;
    int *h_csrRowPtrL = NULL; // <int> n+1
    int *h_csrColIndL = NULL; // <int> nnzL
    double *h_csrValL = NULL; // <double> nnzL

    int nnzU = 0;
    int *h_csrRowPtrU = NULL; // <int> n+1
    int *h_csrColIndU = NULL; // <int> nnzU
    double *h_csrValU = NULL; // <double> nnzU

    int *h_P = NULL; // <int> n, P = Plu * Qreorder
    int *h_Q = NULL; // <int> n, Q = Qlu * Qreorder

    int *d_csrRowPtrA = NULL; // <int> n+1
    int *d_csrColIndA = NULL; // <int> nnzA
    double *d_csrValA = NULL; // <double> nnzA
    
    //array of pointers to the values of each matrix in the batch (of size
    //batchSize) on the device
    double **d_A_array=NULL;
    //For example, if d_A_batch is the array (of size batchSize*nnzA) containing 
    //the values of each matrix in the batch written contiguosly one matrix  
    //after another on the device, then d_A_array[j] = &d_A_batch[nnzA*j];
    //for j=0,...,batchSize-1.
    double *d_A_batch=NULL; 

    double *d_x = NULL; // <double> n, x = A \ b 
    double *d_b = NULL; // <double> n, a copy of h_b
    double *d_r = NULL; // <double> n, r = b - A*x

    //array (of size batchSize*n*nrhs) containing the values of each rhs in 
    //the batch written contiguously one rhs after another on the device
    double *d_X_batch = NULL;
    double **d_X_array = NULL;

    int *d_P = NULL; // <int> n, P*A*Q^T = L*U
    int *d_Q = NULL; // <int> n 
  
    double *d_T = NULL; // working space in cusolverRfSolve
                        // |d_T| = 2*batchSize*n*nrhs

    // the constants used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;
    // the constants used in cusolverRf
    // nzero is the value below which zero pivot is flagged.
    // nboost is the value which is substitured for zero pivot.
    double nzero = 0.0;
    double nboost= 0.0;
    // the constant used in cusolverSp
    // singularity is -1 if A is invertible under tol
    // tol determines the condition of singularity
    // pivot_threshold decides pivoting strategy            
    int singularity = 0; 
    const double tol = 1.e-14;
    const double pivot_threshold = 1.0;
    // the constants used in cusolverRf
    const cusolverRfFactorization_t fact_alg = CUSOLVERRF_FACTORIZATION_ALG0; // default
    const cusolverRfTriangularSolve_t solve_alg = CUSOLVERRF_TRIANGULAR_SOLVE_ALG1; // default

    double x_inf = 0.0; // |x|
    double r_inf = 0.0; // |r|
    double A_inf = 0.0; // |A|
    //int errors = 0;

    double start, stop;
    double time_reorder;
    double time_perm;
    double time_sp_analysis;
    double time_sp_factor;
    double time_sp_solve;
    double time_sp_extract;
    double time_rf_assemble;
    double time_rf_reset;
    double time_rf_refactor;
    double time_rf_solve;

    printf("step 1.1: preparation\n");
    printf("step 1.1: read matrix market format\n");

    if ( rowsA != colsA )
    {
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }

    printf("WARNING: cusolverRf only works for base-0 \n");
    if (baseA)
    {
        for(int i = 0 ; i <= rowsA ; i++)
        {
            h_csrRowPtrA[i]--;
        }
        for(int i = 0 ; i < nnzA ; i++)
        {
            h_csrColIndA[i]--;
        }
        baseA = 0;
    }

    N = rowsA;
    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    // setup batch of A
    h_A_array = (double**)malloc(sizeof(double*)*batchSize);
    h_A_batch = (double*)malloc(sizeof(double)*batchSize*nnzA);
    for (int i = 0; i < batchSize; ++i)
    {
        memcpy(&h_A_batch[i*nnzA], &h_csrValA[i*nnzA], sizeof(double)*nnzA);
    }

    checkCudaErrors(cusolverSpCreate(&cusolverSpH));
    checkCudaErrors(cusparseCreate(&cusparseH));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverSpSetStream(cusolverSpH, stream));
    checkCudaErrors(cusparseSetStream(cusparseH, stream));

    checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));

    if (baseA) 
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }

    h_Qreorder   = (int*)malloc(sizeof(int)*colsA);

    h_csrRowPtrB = (int*   )malloc(sizeof(int)*(rowsA+1));
    h_csrColIndB = (int*   )malloc(sizeof(int)*nnzA);
    h_csrValB    = (double*)malloc(sizeof(double)*nnzA);
    h_mapBfromA  = (int*   )malloc(sizeof(int)*nnzA);

    h_x    = (double*)malloc(sizeof(double)*colsA);
    h_X_array = (double**)malloc(sizeof(double*)*batchSize);
    h_X_batch = (double*)malloc(sizeof(double)*batchSize*N);
    h_b    = (double*)malloc(sizeof(double)*rowsA);
    h_r    = (double*)malloc(sizeof(double)*rowsA);
    h_xhat = (double*)malloc(sizeof(double)*colsA);
    h_bhat = (double*)malloc(sizeof(double)*rowsA);

    assert(NULL != h_Qreorder);

    assert(NULL != h_csrRowPtrB);
    assert(NULL != h_csrColIndB);
    assert(NULL != h_csrValB   );
    assert(NULL != h_mapBfromA);

    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);
    assert(NULL != h_xhat);
    assert(NULL != h_bhat);

    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA   , sizeof(double)*nnzA));
    checkCudaErrors(cudaMalloc((void **)&d_A_array   , sizeof(double*)*batchSize));
    checkCudaErrors(cudaMalloc((void **)&d_A_batch   , sizeof(double)*batchSize*nnzA));
    for (int i = 0; i < batchSize; ++i)
    {
        h_A_array[i] = &(d_A_batch[i*nnzA]);
    }
    checkCudaErrors(cudaMemcpy(d_A_array, h_A_array, batchSize * sizeof(double*), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_X_array, sizeof(double*)*batchSize));
    checkCudaErrors(cudaMalloc((void **)&d_X_batch, sizeof(double)*batchSize*N));
    for (int i = 0; i < batchSize; ++i)
    {
        h_X_array[i] = &(d_X_batch[i*rowsA]);
    }
    checkCudaErrors(cudaMemcpy(d_X_array, h_X_array, batchSize * sizeof(double*), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_P, sizeof(int)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_Q, sizeof(int)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_T, sizeof(double)*rowsA*2*batchSize));



    /***/
    
    
    /***/
    printf("step 1.2: set random right hand side vector (b) in range -1 to 1\n");
    for(int row = 0 ; row < rowsA ; row++){
        h_b[row] = b[row];
    }
    srand(time(NULL));
    for(int i = 0; i < batchSize*colsA; ++i)
    {
        h_X_batch[i] = b[i];//(double)rand()/RAND_MAX*2.0-1.0; + 0.0001 * i/colsA
    }
    
    printf("step 2: reorder the matrix to reduce zero fill-in\n");
    printf("        Q = symrcm(A) or Q = symamd(A) \n");
    start = second();
    start = second();

    if ( 0 == strcmp(opts.reorder, "symrcm") )
    {
        checkCudaErrors(cusolverSpXcsrsymrcmHost(
            cusolverSpH, rowsA, nnzA,
            descrA, h_csrRowPtrA, h_csrColIndA, 
            h_Qreorder));
    }
    else if ( 0 == strcmp(opts.reorder, "symamd") )
    {
        checkCudaErrors(cusolverSpXcsrsymamdHost(
            cusolverSpH, rowsA, nnzA,
            descrA, h_csrRowPtrA, h_csrColIndA, 
            h_Qreorder));
    }
    else 
    {
        fprintf(stderr, "Error: %s is unknow reordering\n", opts.reorder);
        return 1;
    }

    stop = second();
    time_reorder = stop - start;

    printf("step 3: B = Q*A*Q^T\n");
    memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int)*(rowsA+1));
    memcpy(h_csrColIndB, h_csrColIndA, sizeof(int)*nnzA);
    
    start = second();
    start = second();

    checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrRowPtrB, h_csrColIndB,
        h_Qreorder, h_Qreorder,
        &size_perm));

    if (buffer_cpu) {
        free(buffer_cpu); 
    }
    buffer_cpu = (void*)malloc(sizeof(char)*size_perm);
    assert(NULL != buffer_cpu);

    // h_mapBfromA = Identity 
    for(int j = 0 ; j < nnzA ; j++){
        h_mapBfromA[j] = j;
    }
    checkCudaErrors(cusolverSpXcsrpermHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrRowPtrB, h_csrColIndB,
        h_Qreorder, h_Qreorder,
        h_mapBfromA,
        buffer_cpu));

    // B = A( mapBfromA )
    for(int j = 0 ; j < nnzA ; j++){
        h_csrValB[j] = h_csrValA[ h_mapBfromA[j] ];
    }

    stop = second();
    time_perm = stop - start;

    printf("step 4: solve A*x = b by LU(B) in cusolverSp\n");

    printf("step 4.1: create opaque info structure\n");
    checkCudaErrors(cusolverSpCreateCsrluInfoHost(&info));

    printf("step 4.2: analyze LU(B) to know structure of Q and R, and upper bound for nnz(L+U)\n");
    start = second();
    start = second();

    checkCudaErrors(cusolverSpXcsrluAnalysisHost(
        cusolverSpH, rowsA, nnzA,
        descrA, h_csrRowPtrB, h_csrColIndB,
        info));

    stop = second();
    time_sp_analysis = stop - start;

    printf("step 4.3: workspace for LU(B)\n");
    checkCudaErrors(cusolverSpDcsrluBufferInfoHost(
        cusolverSpH, rowsA, nnzA,
        descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        info,
        &size_internal,
        &size_lu));

    if (buffer_cpu) { 
        free(buffer_cpu); 
    }
    buffer_cpu = (void*)malloc(sizeof(char)*size_lu);
    assert(NULL != buffer_cpu);

    printf("step 4.4: compute Ppivot*B = L*U \n");
    start = second();
    start = second();

    checkCudaErrors(cusolverSpDcsrluFactorHost(
        cusolverSpH, rowsA, nnzA,
        descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
        info, pivot_threshold,
        buffer_cpu));

    stop = second();
    time_sp_factor = stop - start;

    // TODO: check singularity by tol
    printf("step 4.5: check if the matrix is singular \n");
    checkCudaErrors(cusolverSpDcsrluZeroPivotHost(
        cusolverSpH, info, tol, &singularity));

    if ( 0 <= singularity){
        fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        return 1;
    }


    printf("step 4.6: solve A*x = b \n");
    printf("    i.e.  solve B*(Qx) = Q*b \n");
    start = second();
    start = second();

    // b_hat = Q*b
    for(int j = 0 ; j < rowsA ; j++){
        h_bhat[j] = h_b[h_Qreorder[j]];
    }
    // B*x_hat = b_hat
    checkCudaErrors(cusolverSpDcsrluSolveHost(
        cusolverSpH, rowsA, h_bhat, h_xhat, info, buffer_cpu));

    // x = Q^T * x_hat
    for(int j = 0 ; j < rowsA ; j++){
        h_x[h_Qreorder[j]] = h_xhat[j];
    }

    stop = second();
    time_sp_solve = stop - start;

    printf("step 4.7: evaluate residual r = b - A*x (result on CPU)\n");
    // use GPU gemv to compute r = b - A*x
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_r, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, sizeof(double)*colsA, cudaMemcpyHostToDevice));
    cout << "h_x" << endl;
    //for  (int i = 0; i < colsA; i++)
    //    cout << h_x[i] << " ";
    //cout << endl;
    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseSpMatDescr_t matA = NULL;
    if (baseA)
    {
        checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F));
    }
    else
    {
        checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    }
    //cusparseDcsrmv CUSPARSE_OPERATION_NON_TRANSPOSE
    /*checkCudaErrors(cusparseDbsrmv(cusparseH,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        rowsA,
        colsA,
        nnzA,
        &minus_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_x,
        &one,
        d_r));*/
    cusparseDnVecDescr_t vecx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F));
    size_t bufferSize = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
        &one, vecAx, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    void *buffer = NULL;
    cout << "Buffer Size " << bufferSize*1000 << endl;
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));
    checkCudaErrors(cusparseSpMV(cusparseH,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &minus_one,
            matA,
            vecx,
            &one,
            vecAx, CUDA_R_64F,CUSPARSE_MV_ALG_DEFAULT, buffer));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));
    /*cout << "h_r" << endl;
    for  (int i = 0; i < rowsA; i++)
        cout << h_r[i] << " ";
    cout << endl;*/
    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);

    printf("(CPU) |b - A*x| = %E \n", r_inf);
    printf("(CPU) |A| = %E \n", A_inf);
    printf("(CPU) |x| = %E \n", x_inf);
    printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

    printf("step 5: extract P, Q, L and U from P*B*Q^T = L*U \n");
    printf("        L has implicit unit diagonal\n");
    start = second();
    start = second();

    checkCudaErrors(cusolverSpXcsrluNnzHost(
        cusolverSpH,
        &nnzL,
        &nnzU,
        info));

    h_Plu = (int*)malloc(sizeof(int)*rowsA);
    h_Qlu = (int*)malloc(sizeof(int)*colsA);

    h_csrValL    = (double*)malloc(sizeof(double)*nnzL);
    h_csrRowPtrL = (int*)malloc(sizeof(int)*(rowsA+1)); 
    h_csrColIndL = (int*)malloc(sizeof(int)*nnzL);

    h_csrValU    = (double*)malloc(sizeof(double)*nnzU);
    h_csrRowPtrU = (int*)malloc(sizeof(int)*(rowsA+1)); 
    h_csrColIndU = (int*)malloc(sizeof(int)*nnzU);

    assert(NULL != h_Plu);
    assert(NULL != h_Qlu);

    assert(NULL != h_csrValL);
    assert(NULL != h_csrRowPtrL);
    assert(NULL != h_csrColIndL);

    assert(NULL != h_csrValU);
    assert(NULL != h_csrRowPtrU);
    assert(NULL != h_csrColIndU);

    checkCudaErrors(cusolverSpDcsrluExtractHost(
        cusolverSpH,
        h_Plu,
        h_Qlu,
        descrA,
        h_csrValL, 
        h_csrRowPtrL,
        h_csrColIndL,
        descrA,
        h_csrValU,
        h_csrRowPtrU,
        h_csrColIndU,
        info,
        buffer_cpu));

    stop = second();
    time_sp_extract = stop - start;

    printf("nnzL = %d, nnzU = %d\n", nnzL, nnzU);

/*  B = Qreorder*A*Qreorder^T
 *  Plu*B*Qlu^T = L*U
 *
 *  (Plu*Qreorder)*A*(Qlu*Qreorder)^T = L*U
 *  
 *  Let P = Plu*Qreroder, Q = Qlu*Qreorder, 
 *  then we have
 *      P*A*Q^T = L*U
 *  which is the fundamental relation in cusolverRf.
 */
    printf("step 6: form P*A*Q^T = L*U\n");

    h_P = (int*)malloc(sizeof(int)*rowsA);
    h_Q = (int*)malloc(sizeof(int)*colsA);
    assert(NULL != h_P);
    assert(NULL != h_Q);

    printf("step 6.1: P = Plu*Qreroder\n");
    // gather operation, P = Qreorder(Plu)
    for(int j = 0 ; j < rowsA ; j++){
        h_P[j] = h_Qreorder[h_Plu[j]];
    }

    printf("step 6.2: Q = Qlu*Qreorder \n");
    // gather operation, Q = Qreorder(Qlu)
    for(int j = 0 ; j < colsA ; j++){
        h_Q[j] = h_Qreorder[h_Qlu[j]];
    }

    printf("step 7: create cusolverRf handle\n");
    checkCudaErrors(cusolverRfCreate(&cusolverRfH));

    printf("step 8: set parameters for cusolverRf \n");
    // numerical values for checking "zeros" and for boosting.
    checkCudaErrors(cusolverRfSetNumericProperties(cusolverRfH, nzero, nboost));

    // choose algorithm for refactorization and solve
    checkCudaErrors(cusolverRfSetAlgs(cusolverRfH, fact_alg, solve_alg));

    // matrix mode: L and U are CSR format, and L has implicit unit diagonal
    checkCudaErrors(cusolverRfSetMatrixFormat(
        cusolverRfH, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));

    // fast mode for matrix assembling
    checkCudaErrors(cusolverRfSetResetValuesFastMode(
        cusolverRfH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));

    printf("step 9: assemble P*A*Q = L*U \n");
    start = second();
    start = second();

    for (int i = 0; i < batchSize; ++i)
    {
        h_A_array[i] = &(h_A_batch[batchSize*i]);
    }
    checkCudaErrors(cusolverRfBatchSetupHost(
        batchSize,
        rowsA, nnzA, 
        h_csrRowPtrA, h_csrColIndA, h_A_array,
        nnzL, 
        h_csrRowPtrL, h_csrColIndL, h_csrValL, 
        nnzU, 
        h_csrRowPtrU, h_csrColIndU, h_csrValU, 
        h_P, 
        h_Q, 
        cusolverRfH));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_rf_assemble = stop - start;

    printf("step 10: analyze to extract parallelism \n");
    checkCudaErrors(cusolverRfBatchAnalyze(cusolverRfH));

    printf("step 11: import A to cusolverRf \n");
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA     , cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA   , h_csrValA   , sizeof(double)*nnzA  , cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_batch   , h_A_batch  , sizeof(double)*batchSize*nnzA  , cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_P, h_P, sizeof(int)*rowsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Q, h_Q, sizeof(int)*colsA, cudaMemcpyHostToDevice));

    start = second();
    start = second();

    checkCudaErrors(cusolverRfBatchResetValues(
        batchSize,
        rowsA,nnzA,
        d_csrRowPtrA, d_csrColIndA, d_A_array,
        d_P,
        d_Q,
        cusolverRfH));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_rf_reset = stop - start;

    printf("step 12: refactorization \n");
    start = second();
    start = second();

    checkCudaErrors(cusolverRfBatchRefactor(cusolverRfH));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_rf_refactor = stop - start;

    printf("step 13: solve A*x = b \n");
    //checkCudaErrors(cudaMemcpy(d_x, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_X_batch, h_X_batch, sizeof(double)*batchSize*rowsA, cudaMemcpyHostToDevice));

    start = second();
    start = second();

    checkCudaErrors(cusolverRfBatchSolve(cusolverRfH, d_P, d_Q, 1, d_T, rowsA, d_X_array, rowsA));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_rf_solve = stop - start;

    printf("step 14: evaluate residual r = b - A*x (result on GPU)\n");
    //checkCudaErrors(cudaMemcpy(d_r, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
    //size_t bufferSize = 0;
    
    
    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
        &one, vecAx, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &bufferSize));
    //void *buffer = NULL;
    cout << "Buffer Size " << bufferSize*1000 << endl;
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));
    //checkCudaErrors(cudaMemcpy(h_X_batch, d_X_batch, sizeof(double)*batchSize*rowsA, cudaMemcpyDeviceToHost));


for (int i=0; i < batchSize; ++i)
    {
        checkCudaErrors(cudaMemcpy(d_r, &h_X_batch[i*colsA], sizeof(double)*rowsA, cudaMemcpyHostToDevice));
        checkCudaErrors(cusparseCreateDnVec(&vecx, N, &d_X_batch[i*colsA], CUDA_R_64F));
        // todo: cusparseSpMM
        
        checkCudaErrors(cusparseSpMV(cusparseH,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &minus_one,
            matA,
            vecx,
            &one,
            vecAx, CUDA_R_64F,CUSPARSE_CSRMV_ALG1, buffer));

        checkCudaErrors(cudaMemcpy(h_x, &d_X_batch[i*colsA], sizeof(double)*colsA, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));
        memcpy(&x[i*colsA], h_x, sizeof(double)*colsA);
        //for (int i = 0; i < rowsA; i++)
            //cout << h_x[i] << " " << h_r[i] << endl;
        
        x_inf = vec_norminf(colsA, h_x);
        r_inf = vec_norminf(rowsA, h_r);
        //printf("(GPU) |b - A*x| = %E ", r_inf);
        //printf("(GPU) |A| = %E \n", A_inf);
        //printf("(GPU) |x| = %E \n", x_inf);
        if (i < 10)
        printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));
    }
    
    
    printf("===== statistics \n");
    printf(" nnz(A) = %d, nnz(L+U) = %d, zero fill-in ratio = %f\n", 
        nnzA, nnzL + nnzU, ((double)(nnzL+nnzU))/(double)nnzA);
    printf("\n");
    printf("===== timing profile \n");
    printf(" reorder A   : %f sec\n", time_reorder);
    printf(" B = Q*A*Q^T : %f sec\n", time_perm);
    printf("\n");
    printf(" cusolverSp LU analysis: %f sec\n", time_sp_analysis);
    printf(" cusolverSp LU factor  : %f sec\n", time_sp_factor);
    printf(" cusolverSp LU solve   : %f sec\n", time_sp_solve);
    printf(" cusolverSp LU extract : %f sec\n", time_sp_extract);
    printf("\n");
    printf(" cusolverRf assemble : %f sec\n", time_rf_assemble);
    printf(" cusolverRf reset    : %f sec\n", time_rf_reset);
    printf(" cusolverRf refactor : %f sec\n", time_rf_refactor);
    printf(" cusolverRf solve    : %f sec\n", time_rf_solve/batchSize);

    if (cusolverRfH) { checkCudaErrors(cusolverRfDestroy(cusolverRfH)); }
    if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
    if (cusparseH  ) { checkCudaErrors(cusparseDestroy(cusparseH)); }
    if (stream     ) { checkCudaErrors(cudaStreamDestroy(stream)); }
    if (descrA     ) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
    if (info       ) { checkCudaErrors(cusolverSpDestroyCsrluInfoHost(info)); }
    if (h_Qreorder  ) { free(h_Qreorder); }
    
    if (h_csrRowPtrB) { free(h_csrRowPtrB); }
    if (h_csrColIndB) { free(h_csrColIndB); }
    if (h_csrValB   ) { free(h_csrValB   ); }
    if (h_mapBfromA ) { free(h_mapBfromA ); }

    if (h_x   ) { free(h_x); }
    if (h_b   ) { free(h_b); }
    if (h_r   ) { free(h_r); }
    if (h_xhat) { free(h_xhat); }
    if (h_bhat) { free(h_bhat); }

    if (buffer_cpu) { free(buffer_cpu); }

    if (h_Plu) { free(h_Plu); }
    if (h_Qlu) { free(h_Qlu); }
    if (h_csrRowPtrL) { free(h_csrRowPtrL); }
    if (h_csrColIndL) { free(h_csrColIndL); }
    if (h_csrValL   ) { free(h_csrValL   ); }
    if (h_csrRowPtrU) { free(h_csrRowPtrU); }
    if (h_csrColIndU) { free(h_csrColIndU); }
    if (h_csrValU   ) { free(h_csrValU   ); }

    if (h_P) { free(h_P); }
    if (h_Q) { free(h_Q); }

    if (d_csrValA   ) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }
    if (d_P) { checkCudaErrors(cudaFree(d_P)); }
    if (d_Q) { checkCudaErrors(cudaFree(d_Q)); }
    if (d_T) { checkCudaErrors(cudaFree(d_T)); }

    return 0;
}



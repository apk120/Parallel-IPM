#include <bits/stdc++.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#include <cusolverRf.h>
#include <cuda_runtime.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include "inc/helper_string.h"
#include "inc/helper_cusolver.h"
#include "inc/helper_cuda.h"

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

void printVectorfloat(const float *V, int m);
void sparesify(float *M, int m, int n, float *A, int *IA, int *JA);
int findNNZ(const float *M, int N);
void printMatrix(const float *A, int nr_rows_A, int nr_cols_A);
void printVector(const int *V, int m);
void getMatSize(string fname, int *rows, int *nnz);
void loadMat(float *A, int *IA, int *JA, string fname, int *rows, int *nnz);
void loadVec(float *V, string fname);
void writeVec(float *V, int n, string fname);
void writeVecInt(int *V, int n, string fname);
void createA(float *A, float *csrValA, int *csrRowA, int *csrColA, int m, int n, int nnz);
int solve (int rowsA, int colsA, int N, int nnzA, int batchSize, int baseA, \
            double *h_csrValA, int *h_csrRowPtrA, int *h_csrColIndA, \
             double *x, double *b, struct testOpts opts);
void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts);
/*
* Batched Dot product of 2 Dense Vectors
* Each Block works on a single batch of vectors
* Number of Blocks must be >= Number of Batches (no_batch) 
*/
__global__ void dot(float *x_b, float *s_b, float *mu, int n, int no_batch)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    temp[threadIdx.x] = 0.0;
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    float *x = &x_b[batch* n];
    float *s = &s_b[batch * n];   
    
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
* Multiplication of a CSR Sparse Matrix with a Vector
*/
__global__ void csr_mul_Av(float *csrValA, int *csrRowA,\
             int *csrColA, float *B_b, float *res_b, int m, int n, int nnz, int no_batch)
{  
    int batch;
    //__shared__ float sum[THREADS_PER_BLOCK];
    //__shared__ int i[THREADS_PER_BLOCK];
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else 
        return;

    float *B = &B_b[batch * n];
    float *res = &res_b[batch * m];

    for (int j = threadIdx.x; j < m; j += blockDim.x)
    {
        float sum = 0.0;
        for (int i = csrRowA[j] - 1; i < csrRowA[j + 1] - 1; i++)
        {
            if (i >= nnz)
                printf("wrrong, %d\n", i);
            sum += csrValA[i] * B[csrColA[i] - 1]; //B[]
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
__global__ void csr_diag_matmul(float *csrValA, int *csrRowA, int *csrColA, float *csrValres_b,\
                    float *d_b, float *y_b, int div, int m, int n, int nnz, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;

    float *d = &d_b[batch * n];
    float *y = &y_b[batch * n];
    float *csrValRes = &csrValres_b[batch * nnz];
    float zy = 1.0;
    for (int j = threadIdx.x; j < nnz; j += blockDim.x)
    {
        if (div == 0)
            csrValRes[j] = csrValA[j] * d[csrColA[j] - 1];
        else if (div == 1)
        {
            float sign = 1.0;
            if (y[csrColA[j] - 1] < 0)
                sign = -1.0;
            
            if (abs(y[csrColA[j] - 1]) < 1e-9)
                zy = 1e-9*sign;
            else
                zy = y[csrColA[j] - 1];
            csrValRes[j] = csrValA[j] / zy;
        }
        else
        {
            float sign = 1.0;
            if (y[csrColA[j] - 1] < 0)
                sign = -1.0;
            
            if (abs(y[csrColA[j] - 1]) < 1e-9)
                zy = 1e-9*sign;
            else
                zy = y[csrColA[j] - 1];
            csrValRes[j] = csrValA[j] * (d[csrColA[j] - 1] + 1e-13) / zy;
        }   
    }
}

/*
* Multiplication of Diagonal Matrix with vector 
* Diagonal Matrix is input as a Dense Vector of Diagonal Elements
* type = 0 for d1*x/d2, type = 1 for x/d2
*/
__global__ void diag_vector_mul(float *diag_b1, float *diag_b2, float *x_b, float *res_b, \
                                int n, int type, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else 
        return;
    
    float *diag1 = &diag_b1[n * batch];
    float *diag2 = &diag_b2[n * batch];
    float *x = &x_b[n * batch];
    float *res = &res_b[n * batch];
    for(int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float di;
        float sign = 1.0;
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
__global__ void vector_add(float *x_b, float *y_b, float *z_b, float *res_b, float a1, float a2, \
                        float a3, int n, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else    
        return;
    float *x = &x_b[batch * n];
    float *y = &y_b[batch * n];
    float *z = &z_b[batch * n];
    float *res = &res_b[batch * n];

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        res[i] = a1 * x[i] + a2 * y[i] + a3 * z[i];
}

/*
* Batched Vector Element Wise Multiplication of 
* 2 vectors-> res[i] = a1*mu + a2*x[i]*y[i] 
* Each Block works on a single batch of vectors
* Number of Blocks must be >= Number of Batches (no_batch)
*/
__global__ void vector_mul(float *x_b, float *y_b, float *mu, float *res_b,\
                            float a1, float a2, int n, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    float *x = &x_b[batch * n];
    float *y = &y_b[batch * n];
    float *res = &res_b[batch * n];

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        res[i] = a1 * mu[batch] + a2 * x[i] * y[i]; 
}

/*
*
*/
__global__ void find_update_param(float *x_b, float *s_b, float *ap_aff, \
                    float *ad_aff, float *dx_aff_b, float *ds_aff_b, int n, int no_batch)
{
    int batch;
    if(blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;

    if (threadIdx.x == 0)
    {
        float *x = &x_b[n * batch];
        float *dx_aff = &dx_aff_b[n * batch];
        float *s = &s_b[n * batch];
        float *ds_aff = &ds_aff_b[n * batch];
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
__global__ void update_vars(float *x_b, float *s_b, float *y_b, float *dx_aff_b,\
            float *ds_aff_b, float *dy_aff_b, float *ap_aff, float *ad_aff, \
            int m, int n, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else    
        return;
    
    float *x = &x_b[n * batch];
    float *y = &y_b[m * batch];
    float *s = &s_b[n * batch];
    float *dx_aff = &dx_aff_b[n * batch];
    float *dy_aff = &dy_aff_b[m * batch];
    float *ds_aff = &ds_aff_b[n * batch];

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
__global__ void ADAt(float *csrValA, int *csrRowA, int *csrColA,\
            float *csrValADA_b, int *csrRowADA_b, int *csrColADA_b, float *x_b, \
            float *y_b, int m, int n, int nnz, int nnzAAt, int no_batch)
{
    int batch;
    __shared__ float sum[THREADS_PER_BLOCK];
    __shared__ int k[THREADS_PER_BLOCK];
    __shared__ int j[THREADS_PER_BLOCK];
    __shared__ int a[THREADS_PER_BLOCK];
    __shared__ int b[THREADS_PER_BLOCK];
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    float *x = &x_b[batch * n];
    float *y = &y_b[batch * n];
    int *csrRowADA = &csrRowADA_b[batch * (m+1)];
    int *csrColADA = &csrColADA_b[batch * nnzAAt];
    float *csrValADA = &csrValADA_b[batch * nnzAAt];

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
                        float zy = y[csrColA[a[threadIdx.x]]-1];
                        float sign = 1.0;
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

__global__ void ADAT(float *A, float *csrValAD_b, int *csrRowAD, int *csrColAD,\
            float *csrValADA_b, int *csrRowADA_b, int *csrColADA_b, float *x_b, \
            float *y_b, int m, int n, int nnz, int nnzAAt, int no_batch)
{
    int batch;
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else
        return;
    //float *x = &x_b[batch * n];
    //float *y = &y_b[batch * n];
    int *csrRowADA = &csrRowADA_b[batch * (m+1)];
    int *csrColADA = &csrColADA_b[batch * nnzAAt];
    float *csrValADA = &csrValADA_b[batch * nnzAAt];
    
    float *csrValAD = &csrValAD_b[batch * nnz];

    for (int i = threadIdx.x; i < m; i += blockDim.x)
    {
        for (int j = csrRowADA[i] - 1; j < csrRowADA[i+1] - 1; j++)
        {
            int k = csrColADA[j] - 1;
            float sum = 0.0;
            for (int a = csrRowAD[i]-1; a < csrRowAD[i+1]-1; a++)
            {
                sum += csrValAD[a] * A[k + (csrColAD[a]-1)*m];
            }
            csrValADA[j] = sum;
        }
    }
}

void AAtransposeCPU(float *csrValA, int *csrRowA, int *csrColA,\
            float *csrValADA, int *csrRowADA, int *csrColADA, int m, int n, int nnz)
{
    int nnzRes = 0;
    for (int j = 0; j < m; j++)
    {
        for (int k = 0; k < m; k++)
        {
            float sum = 0.0;
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

/*void initilaize_solver(cusolverSpHandle_t *cusolverH, cusparseMatDescr_t *descrA, \
                        csrqrInfo_t *info)
{
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

    cusolver_status = cusolverSpCreate(cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cusparse_status = cusparseCreateMatDescr(descrA);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
    cusparseSetMatType(*descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(*descrA, CUSPARSE_INDEX_BASE_ONE); // base-1
    cusolver_status = cusolverSpCreateCsrqrInfo(info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
}*/

int main(int argc, char *argv[])
{
    int m = 6590, n = 12879, batches = 5, iter = 3, nnz; // 6590, 12879, 2036312
    float *csrValA,  *csrValADA, *A;//*A
    int *csrColA, *csrRowA, *csrColADA, *csrRowADA;
    int *csrColAT, *csrRowAT;// *csrColAS, *csrRowAS;
    float *csrValAT, *csrValAS, *csrValAD;
    float *x, *y, *s, *c, *b, *dx_aff, *dy_aff, *ds_aff, *rd, *rp, *rc, *v;   
    float *d_i1, *d_i2, *d_i3, *d_i4;
    //float sigma = 0.8;
    float *ap_aff, *ad_aff, *mu_aff, *mu, *cost;
    int nnz_aat = 2036312;
    cudaDeviceSynchronize();
    cusolverSpHandle_t cusolverH = NULL;
    // GPU does batch QR
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL; // working space for numerical factorization
    cudaError_t cudaStat1 = cudaSuccess;
    struct testOpts opts;
    parseCommandLineArguments(argc, argv, opts);
    findCudaDevice(argc, (const char **)argv);
    // Initialize A
    /*A = (float*)malloc(m* n * sizeof(float));
    for (int i = 0; i < m*n; i++)
        A[i] = 0;
    A[0] = 1, A[3] = 1, A[4] = 1, A[7] = 1;
    nnz = findNNZ(A, m*n);*/

    string fname = "A_sparse.csv";
    getMatSize(fname, &m, &nnz);
    printf("Non Zero Elements: %d, rows: %d\n", nnz, m);

    cudaMallocManaged(&A, sizeof(float)*m*n);
    cudaMallocManaged(&csrValA, sizeof(float)*nnz);
    cudaMallocManaged(&csrRowA, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColA, sizeof(int)*nnz);
    cudaMallocManaged(&csrValAT, sizeof(float)*nnz);
    cudaMallocManaged(&csrRowAT, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColAT, sizeof(int)*nnz);
    cudaMallocManaged(&csrValAS, sizeof(float)*nnz*batches);
    cudaMallocManaged(&csrValAD, sizeof(float)*nnz*batches);
    cudaMallocManaged(&csrValADA, sizeof(float)*nnz_aat*batches);
    cudaMallocManaged(&csrRowADA, sizeof(int)*(m+1)*batches);
    cudaMallocManaged(&csrColADA, sizeof(int)*nnz_aat*batches);
    cudaMallocManaged(&x, sizeof(float)* n * batches);
    cudaMallocManaged(&s, sizeof(float)* n * batches);
    cudaMallocManaged(&y, sizeof(float)* m * batches);
    cudaMallocManaged(&c, sizeof(float)* n * batches);
    cudaMallocManaged(&b, sizeof(float)* m * batches);
    cudaMallocManaged(&dx_aff, sizeof(float) * n *batches);
    cudaMallocManaged(&dy_aff, sizeof(float) * m * batches);
    cudaMallocManaged(&d_i1, sizeof(float) * m * batches);
    cudaMallocManaged(&d_i2, sizeof(float) * m * batches);
    cudaMallocManaged(&d_i3, sizeof(float) * m * batches);
    cudaMallocManaged(&ds_aff, sizeof(float) * n * batches);
    cudaMallocManaged(&d_i4, sizeof(float) * n * batches);
    cudaMallocManaged(&rd, sizeof(float)* n * batches);
    cudaMallocManaged(&rp, sizeof(float)* m * batches);
    cudaMallocManaged(&rc, sizeof(float)* n * batches);
    cudaMallocManaged(&v, sizeof(float)* n * batches);
    cudaMallocManaged(&ap_aff, sizeof(float) * batches);
    cudaMallocManaged(&ad_aff, sizeof(float) * batches);
    cudaMallocManaged(&mu_aff, sizeof(float) * batches);
    cudaMallocManaged(&mu, sizeof(float) * batches);
    cudaMallocManaged(&cost, sizeof(float) * batches);
    CUDIE0();
    /*Initialization of the Problem*/
    float *ib, *ic, *ix, *is, *iy;
    ib = (float *)malloc(m * sizeof(float));
    ic = (float *)malloc(n * sizeof(float));
    ix = (float *)malloc(n * sizeof(float));
    is = (float *)malloc(n * sizeof(float));
    iy = (float *)malloc(m * sizeof(float));
    loadVec(ib, "Btxt.csv");
    loadVec(ic, "Ctxt.csv");
    loadVec(ix, "x_initial.csv");
    loadVec(is, "s_initial.csv");
    loadVec(iy, "y_initial.csv");
    //float ib[2] = {1, 2}, ic[4] = {-1, -1, 1, 1};
    //float ix[4] = {0.8, 0.1, 0.2, 1.9}, is[4] = {0.1, 0.2, 2.1, 2.2}, iy[2] = {-1.1, -1.2};
    
    float miny = 100, maxy = -100;
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
    cout << b[0] << " " << b[6590] << endl;
    cout << y[0] << " " << y[6590] << endl;
    cout << miny << " " << maxy << endl;      
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < n; j++) {
            c[n*i + j] = ic[j];
            x[n*i + j] = ix[j];
            s[n*i + j] = is[j];
        }
    }
    cout << c[0] << " " << c[12578] << endl;
    cout << x[0] << " " << x[12578] << endl;
    cout << s[0] << " " << s[12578] << endl;
    loadMat(csrValA, csrRowA, csrColA, "A_sparse.csv", &m, &nnz);
    /*sparesify(A, m, n, csrValA, csrRowA, csrColA);
    for (int i = 0; i < m*n; i++)
        A[i] = 0;
    A[0] = 1, A[2] = 1, A[5] = 1, A[7] = 1;*/
    loadMat(csrValAT, csrRowAT, csrColAT, "At_sparse.csv", &n, &nnz);
    printVector(csrColAT, 10);
    printVectorfloat(csrValAT, 10);
    printVector(csrRowAT, 10);
    writeVecInt(csrColAT, 1000, "csrColAt_bef.txt");
    writeVecInt(csrRowAT, 1000, "csrRowAt_bef.txt");
    writeVec(csrValAT, 1000, "csrValAt_bef.txt");
    //sparesify(A, n, m, csrValAT, csrRowAT, csrColAT);
    int nnzAAt = nnz_aat, r_aat;
    cout << "Transpose Start" << endl;
    loadMat(csrValADA, csrRowADA, csrColADA, "AAt_sparse.csv", &r_aat, &nnzAAt);
    for (int i = 1; i < batches; i++)
    {
        //AAtransposeCPU(csrValA, csrRowA, csrColA, &csrValADA[nnzAAt*i],\
                    &csrRowADA[(m+1)*i], &csrColADA[nnzAAt*i], m, n, nnz);
        for (int j = 0; j < (m + 1); j++)
        {
            csrRowADA[(m+1)*i + j] = csrRowADA[j];
        }
        for (int j = 0; j < nnzAAt; j++)
        {
            csrColADA[nnzAAt*i + j] = csrColADA[j];
            csrValADA[nnzAAt*i + j] = csrValADA[j];
        }
    }   
    cout << csrRowADA[1] << " " << csrRowADA[m+1] << endl;
    cout << csrValADA[1] <<  " " << csrValADA[nnzAAt+1] << endl;
    cout << csrColADA[1] <<  " " << csrColADA[nnzAAt+1] << endl; 
    cout << "Transpose End" << endl;
    CUDIE0();
    createA(A, csrValA, csrRowA, csrColA, m, n, nnz);
    /*printVectorfloat(csrValADA, nnzAAt * batches);
    printVector(csrRowADA, (m + 1)*batches);
    printVector(csrColADA, nnzAAt * batches);*/

    /*---Initialization*/

    //printVectorfloat(x, n*batches);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;
    cudaEventRecord(start);
    /*Main Loop for IPM iterations*/
    for (int i = 0; i < iter; i++)
    {
        dot<<<batches, THREADS_PER_BLOCK>>>(x, s, mu, n, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        cout << "MU" << endl;
        for (int j = 0; j < batches; j++)
            cout << mu[j]*n << " ";
        cout << endl;
        
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, y, rd,\
                                                    n, m, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        cout << "xxxxxx" << endl;
        printVector(csrColAT, 10);
        printVectorfloat(csrValAT, 10);
        printVector(csrRowAT, 10);
        printVectorfloat(rd, 10);
        writeVecInt(csrColAT, 1000, "csrColAt.txt");
        writeVecInt(csrRowAT, 1000, "csrRowAt.txt");
        writeVec(csrValAT, 1000, "csrValAt.txt");
        //writeVec(rd, n, "yuss2.txt");
        
        vector_add<<<batches, THREADS_PER_BLOCK>>>(c, s, rd, rd, 1, -1, -1, n, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        writeVec(rd, n, "yuss2.txt");
        
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA, x, rp,\
                                                    m, n, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(b, rp, rp, rp, 1, -1, 0, m, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        writeVec(rp, m, "yuss1.txt");
        vector_mul<<<batches, THREADS_PER_BLOCK>>>(x, s, mu, rc, 0, -1, n, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        
        csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
                                 csrValAS, s, s, 1, m, n, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAS, csrRowA, csrColA, rc, d_i1,\
                                                    m, n, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
                                 csrValAD, x, s, 2, m, n, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        cout << "ADAT Start\n";
        ADAT<<<batches, THREADS_PER_BLOCK>>>(A, csrValAD, csrRowA, csrColA, csrValADA,\
                                csrRowADA, csrColADA, x, s, m, n, nnz, nnzAAt, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        writeVec(csrValADA, nnzAAt, "csrValADA.txt");
        writeVecInt(csrColADA, nnzAAt, "csrColADA.txt");
        writeVecInt(csrRowADA, m + 1, "csrRowADA.txt");
        cout << "ADAT End\n";
        //return 0;
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAD, csrRowA, csrColA, rd, d_i2,\
                                                    m, n, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(d_i1, d_i2, rp, d_i3, -1, 1, 1, m, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        
        writeVec(d_i2, m, "yuss3.txt");
        writeVec(d_i3, m, "yuss.txt");
        cout << m << " " << n << " " << nnzAAt<< " " << endl;
        //Solve for dy_aff
        // step 2: create cusolver handle, qr info and matrix descriptor
        cusolver_status = cusolverSpCreate(&cusolverH);
        assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
        cusparse_status = cusparseCreateMatDescr(&descrA);
        assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1
        cusolver_status = cusolverSpCreateCsrqrInfo(&info);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        // step 4: symbolic analysis
        cusolver_status = cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzAAt, \
        descrA, csrRowADA, csrColADA,info);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        // step 5: prepare working space
        cusolver_status = cusolverSpScsrqrBufferInfoBatched(cusolverH, m, m, nnzAAt,\
        descrA, csrValADA, csrRowADA, csrColADA,\
        batches,\
        info,\
        &size_internal,\
        &size_qr);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        printf("numerical factorization needs internal data %lld bytes\n",(long long)size_internal);
        printf("numerical factorization needs working space %lld bytes\n",(long long)size_qr);
        cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
        assert(cudaStat1 == cudaSuccess);
        // step 6: numerical factorization
        // assume device memory is big enough to compute all matrices.
        cusolver_status = cusolverSpScsrqrsvBatched(cusolverH, m, m, nnzAAt,\
        descrA, csrValADA, csrRowADA, csrColADA,\
        d_i3, dy_aff,\
        batches,\
        info,\
        buffer_qr);
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA))? 1:0 ;
        cudaDeviceSynchronize();
        for(int batchId = 0 ; batchId < batches; batchId++){
            // measure |bj - Aj*xj|
            float *csrValAj = csrValADA + batchId * nnzAAt;
            float *xj = dy_aff + batchId * m;
            float *bj = d_i3 + batchId * m;
            // sup| bj - Aj*xj|
            float sup_res = 0;
            for(int row = 0 ; row < m ; row++){
                const int start = csrRowADA[row ] - baseA;
                const int end = csrRowADA[row+1] - baseA;
                float Ax = 0.0; // Aj(row,:)*xj
                for(int colidx = start ; colidx < end ; colidx++){
                    const int col = csrColADA[colidx] - baseA;
                    const float Areg = csrValAj[colidx];
                    const float xreg = xj[col];
                    Ax = Ax + Areg * xreg;
                }
                float r = bj[row] - Ax;
                sup_res = (sup_res > fabs(r))? sup_res : fabs(r);
            }
            printf("batchId %d: sup|bj - Aj*xj| = %E \n", batchId, sup_res);
        }
        for(int batchId = 0 ; batchId < batches; batchId++){
            float *xj = dy_aff + batchId * m;
            for(int row = 0 ; row < min(10, m) ; row++){
                printf("x%d[%d] = %E\n", batchId, row, xj[row]);
            }
            printf("\n");
        }

        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, dy_aff, ds_aff,\
                                                    n, m, nnz, batches);
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
        update_vars<<<batches, THREADS_PER_BLOCK>>>(x, s, y, dx_aff, ds_aff, dy_aff, ap_aff,\
                                ad_aff, m, n, batches);
        cudaDeviceSynchronize();
        writeVec(x, n, "x_upd.txt");
        writeVec(s, n, "s_upd.txt");
        cout << "Cost: " << endl;
        dot<<<batches, THREADS_PER_BLOCK>>>(c, x, cost, n, batches);
        cudaDeviceSynchronize();
        for (int j = 0; j < batches; j++)
            cout << cost[j]*n << " ";
        cout << endl;
    }
    //for (int i  = 0; i < batches; i++)
      //  cout << mu[i] << endl;
    /*for (int i = 0; i < batches; i++){
        for (int j = 0; j < n; j++)
            cout << rd[i*n + j] << " ";
        cout << endl;
    }
    printVectorfloat(csrValADA, 2*batches);
    printVectorfloat(d_i1, m*batches);
    printVectorfloat(d_i2, m*batches);
    printVectorfloat(d_i3, m*batches);
    printVectorfloat(x, n * batches);*/
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


void printVectorfloat(const float *V, int m)
{
    for (int i = 0; i < m; i++)
        std::cout << V[i] << " ";
    std::cout << std::endl;
}
// Generate the three vectors A, IA, JA 
void sparesify(float *M, int m, int n, float *A, int *IA, int *JA)
{
    //int m = M.size(), n = M[0].size();
    int i, j;
    //vi A;
    IA[0] = 1; // IA matrix has N+1 rows
    //vi JA;
    int NNZ = 0;
  
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (M[i + m*j] != 0) {
                A[NNZ] = M[i + m*j];
                JA[NNZ] = j + 1;
  
                // Count Number of Non Zero 
                // Elements in row i
                NNZ++;
            }
        }
        IA[i + 1] = NNZ + 1;
    }
  
    printMatrix(M, m, n);
    printVectorfloat(A, NNZ);
    printVector(IA, m + 1);
    printVector(JA, NNZ);
}
//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
 void printMatrix(const float *A, int nr_rows_A, int nr_cols_A) {
 
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
int findNNZ(const float *M, int N)
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
void loadMat(float *A, int *IA, int *JA, string fname, int *rows, int *nnz)
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
        IA[n] = i;
        n++;
    }
    *rows = n - 1;
    getline(infile, line);
    n = 0;
    stringstream stream1(line);
    while (stream1 >> i)
    {
        JA[n] = i;
        n++;
        //if (n < 50)
            //cout << i << " ";
    }
    cout << endl;
    *nnz = n;
    getline(infile, line);
    n = 0;
    float f;
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
void loadVec(float *V, string fname)
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
        float f;
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

void writeVec(float *V, int n, string fname)
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

void createA(float *A, float *csrValA, int *csrRowA, int *csrColA, int m, int n, int nnz)
{
    for (int i  = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i + m*j] = 0;

    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowA[i] - 1; j < csrRowA[i+1] - 1; j++)
        {
            A[i + m*(csrColA[j]-1)] = csrValA[j];
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



template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

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
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    const float *x,
    float *r_nrminf_ptr)
{
    const int base = (cusparseGetMatIndexBase(descrA) != CUSPARSE_INDEX_BASE_ONE)? 0:1 ;
    const int lower = (CUSPARSE_FILL_MODE_LOWER == cusparseGetMatFillMode(descrA))? 1:0;
    const int unit  = (CUSPARSE_DIAG_TYPE_UNIT == cusparseGetMatDiagType(descrA))? 1:0;

    float r_nrminf = 0;
    for(int row = 0 ; row < n ; row++){
        const int start = csrRowPtr[row]   - base;
        const int end   = csrRowPtr[row+1] - base;
        float dot = 0;
        for(int colidx = start ; colidx < end; colidx++){
            const int col = csrColInd[colidx] - base;
            float Aij = csrVal[colidx];
            float xj  = x[col];
            if ( (row == col) && unit ){
                Aij = 1.0;
            }
            int valid = (row >= col) && lower ||
                        (row <= col) && !lower ;
            if ( valid ){
                dot += Aij*xj;
            }
        }
        float ri = b[row] - dot;
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
    int errors = 0;

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
        h_X_batch[i] = b[i];//(double)rand()/RAND_MAX*2.0-1.0;
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
    cout << endl;
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
    cout << "h_r" << endl;
    for  (int i = 0; i < rowsA; i++)
        cout << h_r[i] << " ";
    cout << endl;
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
        for (int i = 0; i < rowsA; i++)
            cout << h_x[i] << " " << h_r[i] << endl;
        
        x_inf = vec_norminf(colsA, h_x);
        r_inf = vec_norminf(rowsA, h_r);
        //printf("(GPU) |b - A*x| = %E ", r_inf);
        //printf("(GPU) |A| = %E \n", A_inf);
        //printf("(GPU) |x| = %E \n", x_inf);
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



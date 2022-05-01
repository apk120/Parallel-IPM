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

void printVectorfloat(const float *V, int m);
void sparesify(float *M, int m, int n, float *A, int *IA, int *JA);
int findNNZ(const float *M, int N);
void printMatrix(const float *A, int nr_rows_A, int nr_cols_A);
void printVector(const int *V, int m);
void writeVec(float *V, int n, string fname);    
/*
* Batched Dot product of 2 Dense Vectors
* Each Block works on a single batch of vectors
* Number of Blocks must be >= Number of Batches (no_batch) 
*/
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
    __shared__ float sum[THREADS_PER_BLOCK];
    __shared__ int i[THREADS_PER_BLOCK];
    if (blockIdx.x < no_batch)
        batch = blockIdx.x;
    else 
        return;

    float *B = &B_b[batch * n];
    float *res = &res_b[batch * m];

    for (int j = threadIdx.x; j < m; j += blockDim.x)
    {
        sum[threadIdx.x] = 0.0;
        for (i[threadIdx.x] = csrRowA[j] - 1; i[threadIdx.x] < csrRowA[j + 1] - 1; i[threadIdx.x]++)
        {
            sum[threadIdx.x] += csrValA[i[threadIdx.x]] * B[csrColA[i[threadIdx.x]] - 1]; 
        }     
        res[j] = sum[threadIdx.x];
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
            if (abs(y[csrColA[j] - 1]) < 1e-9)
                zy = 1e-9;
            else
                zy = y[csrColA[j] - 1];
            csrValRes[j] = csrValA[j] / zy;
        }
        else
            csrValRes[j] = csrValA[j] * d[csrColA[j] - 1] / y[csrColA[j] - 1];
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
        if (abs(diag2[i]) < 1e-9)
            di = 1e-9;
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
        for (int j = csrRowADA[i] - 1; j < csrRowADA[i+1] - 1; j++)
        {
            int k = csrColADA[j] - 1;
            float sum = 0.0;
            for (int a = csrRowA[i] - 1; a < csrRowA[i + 1] - 1; a++)
            {
                for (int b = csrRowA[k] - 1; b < csrRowA[k+1]-1; b++)
                {
                    if (csrColA[a] == csrColA[b])
                    {
                        float zy = y[csrColA[a]-1];
                        if (abs(zy) < 1e-9)
                            zy = 1e-9;
                        sum += csrValA[a] * csrValA[b] * x[csrColA[a]-1]/zy;
                    }
                }
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

int main()
{
    int m = 2, n = 4, batches = 3, iter = 10, nnz;
    float *A,  *csrValA,  *csrValADA;
    int *csrColA, *csrRowA, *csrColADA, *csrRowADA;
    int *csrColAT, *csrRowAT;// *csrColAS, *csrRowAS;
    float *csrValAT, *csrValAS, *csrValAD;
    float *x, *y, *s, *c, *b, *dx_aff, *dy_aff, *ds_aff, *rd, *rp, *rc, *v;   
    float *d_i1, *d_i2, *d_i3, *d_i4;
    //float sigma = 0.8;
    float *ap_aff, *ad_aff, *mu_aff, *mu, *cost;

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
    
    // Initialize A
    A = (float*)malloc(m* n * sizeof(float));
    for (int i = 0; i < m*n; i++)
        A[i] = 0;
    A[0] = 1, A[3] = 1, A[4] = 1, A[7] = 1;
    nnz = findNNZ(A, m*n);
    printf("Non Zero Elements: %d\n", nnz);

    cudaMallocManaged(&csrValA, sizeof(float)*nnz);
    cudaMallocManaged(&csrRowA, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColA, sizeof(int)*nnz);
    cudaMallocManaged(&csrValAT, sizeof(float)*nnz);
    cudaMallocManaged(&csrRowAT, sizeof(int)*(m+1));
    cudaMallocManaged(&csrColAT, sizeof(int)*nnz);
    cudaMallocManaged(&csrValAS, sizeof(float)*nnz*batches);
    cudaMallocManaged(&csrValAD, sizeof(float)*nnz*batches);
    cudaMallocManaged(&csrValADA, sizeof(float)*2*batches);
    cudaMallocManaged(&csrRowADA, sizeof(int)*(m+1)*batches);
    cudaMallocManaged(&csrColADA, sizeof(int)*2*batches);
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

    /*Initialization of the Problem*/
    float ib[2] = {1, 2}, ic[4] = {-1, -1, 1, 1};
    float ix[4] = {0.8, 0.1, 0.2, 1.9}, is[4] = {0.1, 0.2, 2.1, 2.2}, iy[2] = {-1.1, -1.2};
    
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < m; j++) {
            b[m*i + j] = ib[j];
            y[m*i + j] = iy[j];
        }
    }        
    for (int i = 0; i < batches; i++) {
        for (int j = 0; j < n; j++) {
            c[n*i + j] = ic[j];
            x[n*i + j] = ix[j];
            s[n*i + j] = is[j];
        }
    }
    sparesify(A, m, n, csrValA, csrRowA, csrColA);
    for (int i = 0; i < m*n; i++)
        A[i] = 0;
    A[0] = 1, A[2] = 1, A[5] = 1, A[7] = 1;
    sparesify(A, n, m, csrValAT, csrRowAT, csrColAT);
    int nnzAAt = 2;
    for (int i = 0; i < batches; i++)
    {
        AAtransposeCPU(csrValA, csrRowA, csrColA, &csrValADA[nnzAAt*i],\
                    &csrRowADA[(m+1)*i], &csrColADA[nnzAAt*i], m, n, nnz);
    }   
    printVectorfloat(csrValADA, nnzAAt * batches);
    printVector(csrRowADA, (m + 1)*batches);
    printVector(csrColADA, nnzAAt * batches);

    /*---Initialization*/

    //printVectorfloat(x, n*batches);
    /*Main Loop for IPM iterations*/
    for (int i = 0; i < iter; i++)
    {
        dot<<<batches, THREADS_PER_BLOCK>>>(x, s, mu, n, batches);
        cudaDeviceSynchronize();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAT, csrRowAT, csrColAT, y, rd,\
                                                    n, m, nnz, batches);
        cudaDeviceSynchronize();
        CUDIE0();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(c, s, rd, rd, 1, -1, -1, n, batches);
        cudaDeviceSynchronize();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA, x, rp,\
                                                    m, n, nnz, batches);
        cudaDeviceSynchronize();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(b, rp, rp, rp, 1, -1, 0, m, batches);
        cudaDeviceSynchronize();
        vector_mul<<<batches, THREADS_PER_BLOCK>>>(x, s, mu, rc, 0, -1, n, batches);
        cudaDeviceSynchronize();
        ADAt<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA, csrValADA,\
                                csrRowADA, csrColADA, x, s, m, n, nnz, 2, batches);
        cudaDeviceSynchronize();
        csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
                                 csrValAS, s, s, 1, m, n, nnz, batches);
        cudaDeviceSynchronize();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAS, csrRowA, csrColA, rc, d_i1,\
                                                    m, n, nnz, batches);
        cudaDeviceSynchronize();
        csr_diag_matmul<<<batches, THREADS_PER_BLOCK>>>(csrValA, csrRowA, csrColA,\
                                 csrValAD, x, s, 2, m, n, nnz, batches);
        cudaDeviceSynchronize();
        csr_mul_Av<<<batches, THREADS_PER_BLOCK>>>(csrValAD, csrRowA, csrColA, rd, d_i2,\
                                                    m, n, nnz, batches);
        cudaDeviceSynchronize();
        vector_add<<<batches, THREADS_PER_BLOCK>>>(d_i1, d_i2, rp, d_i3, -1, 1, 1, m, batches);
        cudaDeviceSynchronize();

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
            for(int row = 0 ; row < m ; row++){
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
    printVectorfloat(d_i3, m*batches);*/
    printVectorfloat(x, n * batches);
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
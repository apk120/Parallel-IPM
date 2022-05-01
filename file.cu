#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace std;

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
        //cout << i;
    }
    //cout << endl;
    *nnz = n;
    getline(infile, line);
    n = 0;
    float f;
    stringstream stream2(line);
    while (stream2 >> f)
    {
        A[n] = f;
        n++;
    }
}
// Utility Function to print A, IA, JA vectors
// with some decoration.
void printVector(const int *V, int m)
{
    for (int i = 0; i < m; i++)
        std::cout << V[i] << " ";
    std::cout << std::endl;
}
void printVectorfloat(const float *V, int m)
{
    for (int i = 0; i < m; i++)
        std::cout << V[i] << " ";
    std::cout << std::endl;
}

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
int main()
{
    int *IA, *JA;
    float *A, *B;
    int rows, nnz;
    string fname = "At_sparse.csv";
    getMatSize(fname, &rows, &nnz);
    cudaMallocManaged(&IA, (rows + 1)*sizeof(int));
    cudaMallocManaged(&B, (rows)*sizeof(int));
    cudaMallocManaged(&JA, nnz*sizeof(int));
    cudaMallocManaged(&A, nnz*sizeof(int));
    loadMat(A, IA, JA, fname, &rows, &nnz);
    loadVec(B, "Btxt.csv");
    //printVector(IA, rows+1);
    //printVector(JA, nnz);
    //printVectorfloat(A, nnz);
    cout << B[0] << " " << B[rows - 1] << endl;
    cout << rows << " " << nnz << endl;
    cout << IA[rows] << endl;   
}
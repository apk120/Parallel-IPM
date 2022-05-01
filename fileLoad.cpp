#include <bits/stdc++.h>

using namespace std;

void loadMat(float *A, int *IA, int *JA, string fname, int *rows, int *nnz)
{
    ifstream infile;
    infile.open(fname);
    string line;
    getline(infile, line);
    int i, n = 0;
    cout << line << endl;
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
        cout << i;
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
int main()
{
    int IA[4], JA[5];
    float A[5];
    int rows, nnz;
    string fname = "test.csv";
    loadMat(A, IA, JA, fname, &rows, &nnz);
    printVector(IA, rows+1);
    printVector(JA, nnz);
    printVectorfloat(A, nnz);   
}
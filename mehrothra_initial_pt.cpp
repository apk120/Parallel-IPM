#include <bits/stdc++.h>
#include "eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;

#define MAX_BUF_SIZE 50000000

void loadData(vector<float> &A, vector<float> &B, vector<float> &C, vector<float> &AAt,  int* row, int* col)
{
    int cols = 0, rows = 0;
    //vector<float> buff(MAX_BUF_SIZE);
    ifstream infile;
    infile.open("Atxt.csv");
    while(!infile.eof())
    {
        string line;
        getline(infile, line);
        int temp_cols = 0;
        stringstream stream(line);
        float f;
        while(stream >> f)
        {
            A[cols*rows + temp_cols++] = f;
        }
        
        if(temp_cols == 0)
            continue;
        
        if (cols == 0)
            cols = temp_cols;
        
        rows++;
    }
    *row = rows;
    *col = cols;
    cout << rows << " " << cols << endl;
    infile.close();


    cols = 0, rows = 0;
    infile.open("Btxt.csv");
    while(!infile.eof())
    {
        string line;
        getline(infile, line);
        int temp_cols = 0;
        stringstream stream(line);
        float f;
        while(stream >> f)
        {
            B[cols*rows + temp_cols++] = f;
        }
        
        if(temp_cols == 0)
            continue;
        
        if (cols == 0)
            cols = temp_cols;
        
        rows++;
    }
    cout << rows << " " << cols << endl;
    infile.close();
    for (int i = 0; i < 10; i++)
        cout << B[i] << endl;

    cols = 0, rows = 0;
    infile.open("Ctxt.csv");
    while(!infile.eof())
    {
        string line;
        getline(infile, line);
        int temp_cols = 0;
        stringstream stream(line);
        float f;
        while(stream >> f)
        {
            C[cols*rows + temp_cols++] = f;
        }
        
        if(temp_cols == 0)
            continue;
        
        if (cols == 0)
            cols = temp_cols;
        
        rows++;
    }
    cout << rows << " " << cols << endl;
    infile.close();

    cols = 0, rows = 0;
    infile.open("AAt.csv");
    while(!infile.eof())
    {
        string line;
        getline(infile, line);
        int temp_cols = 0;
        stringstream stream(line);
        float f;
        while(stream >> f)
        {
            AAt[cols*rows + temp_cols++] = f;
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
    int m, n;
    vector<float> A_x(MAX_BUF_SIZE);
    vector<float> B_x(6600);
    vector<float> C_x(6600);
    vector<float> AAt(MAX_BUF_SIZE);
    loadData(A_x, B_x, C_x, AAt, &m, &n);
    cout << m << " " << n << endl;

    MatrixXd A(m, n), X(n, n), S(n, n), At(n, m), D(n, n), ADA(m, m), AS(m, n);
    VectorXd x(n), y(m), s(n), c(n), b(m), dx_aff(n), dy_aff(m), ds_aff(n), dx_cor(n), dy_cor(m), ds_cor(n);
    VectorXd rd(n), rp(m), rc(n), v(n), e(n), mid(m), x_i(m), s_i(m);
    double mu, sigma = 0.8, ap_aff, ad_aff, mu_aff, ap_k, ad_k;
    int iter = 10;
    double delta_x = 0.0, delta_s = 0.0, i_x, i_y;
    cout << "1 " << endl;
    for (int i = 0; i < m; i++)
    {
        b(i) = B_x[i];
        if (i < 10)
            cout << B_x[i] << endl;
    }
        
    for (int i = 0; i < n; i++)
        c(i) = C_x[i];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            A(i, j) = A_x[n*i + j];
            if (i < 10 && j < 10)
                cout << A(i, j) << " " << A_x[n*i + j];
            if (i < 10 && j < 10)
                cout << endl;
        }
            
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            ADA(i, j) = AAt[n*i + j];
    cout << 1.5 << endl;
    At = A.transpose();
    //ADA = A * A.transpose();
    cout << 1.7 << endl;
    x_i = ADA.inverse()*b;
    cout << 2 << endl;
    s_i = A*c;
    x = A.transpose() * x_i;
    y = ADA.inverse()* s_i;
    for (int i = 0; i < 10; i++)
    {
        cout << "x : " << x_i(i) << endl;
        cout << "y : " << s_i(i) << endl;
        
    }
    s = c - At * y;
    for (int i = 0; i < n; i++)
    {
        delta_x = max(delta_x, -1.5*x(i));
        delta_s = max(delta_s, -1.5*s(i));
    }
    for (int i = 0; i < n; i++)
    {
        e(i) = 1.0;
        x(i) = x(i) + delta_x;
        s(i) = s(i) + delta_s;
    }
    i_x = x.transpose()*s;
    i_y = e.transpose()*s;
    cout << i_x << " " << i_y << endl;
    delta_x = 0.5 * (i_x/i_y);
    delta_s = 0.5 * (i_x/i_y);
    for (int i = 0; i < n; i++)
    {
        x(i) = x(i) + delta_x;
        s(i) = s(i) + delta_s;
    }
    cout << "3 "<< endl;
    mid = b - A * x;
    for (int i = 0;  i < n; i++)
    {
        if (x(i) < 0.0)
            cout << "x : " << x(i) << endl;
        if (s(i) < 0.0)
            cout << "s : " << s(i) << endl;
    }
    for (int i = 0; i < 10; i++)
    {
        cout << "b-Atxs+s: " << mid(i) << endl;
        cout << "x : " << x(i) << endl;
        cout << "s : " << s(i) << endl;
        
    }
}
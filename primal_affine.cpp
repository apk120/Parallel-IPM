#include <bits/stdc++.h>
#include "eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;
 
int main()
{
    clock_t tStart = clock();


    int m, n = 500;
    m = 250;
    MatrixXd A(m, n), D(n, n), B(m, n), Tx(m, m);
    VectorXd b(m), c(n), x(n), p(n), w(n), v(m), g(n), d(n), z(n), e(n), xprev(n), err(n);
    int iter = 10;
    cout << "Enter Iterations:" << endl;
    cin >> iter;
    double lambda = 1.0;
    //b(0) = 1;
    //b(1) = 2;
    //c(0) = -1, c(1) = -1, c(2) = 1, c(3) = 1;

    srand(0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A(i, j) = (rand() % 10) * 1.0;
    //A(0, 0) = 1, A(0, 2) = 1, A(1, 1) = 1, A(1, 3) = 1;
    //x << 0.8, 0.1, 0.2, 1.9;
    for (int i = 0; i < n; i++)
    {
        x(i) = rand() % 10;
        c(i) = rand() % 10;
    }
    b = A * x;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            D(i, j) = 0;

    for (int i = 0; i < n; i++)
        e(i) = 1.0;

    cout << "Cost: " << c.transpose() * x << endl;
    for (int i = 0; i < iter; i++)
    {
        cout << " Iteration: " << i << endl;
        for (int j = 0; j < n; j++)
            D(j, j) = 1.0 * x(j);
        

        /*for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                cout << D(j, k) << " ";
            }
            cout << endl;
        }*/

        B = A * D;
        /*cout << "B " << endl;
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < n; k++)
            {
                cout << B(j, k) << " ";
            }
            cout << endl;
        }*/

        p = D * c;
        /*cout << "p" << endl;
        for (int j = 0; j < n; j++)
            cout << p(j) << " ";
        cout << endl;*/

        w = B * p;
        Tx = B * B.transpose();
        v = Tx.inverse() * w;//Tx.colPivHouseholderQr().solve(w);
        g = B.transpose() * v;
        d = g - p;
        /*cout << "d" << endl;
        for (int j = 0; j < n; j++)
            cout << d(j) << " ";
        cout << endl;*/

        lambda = 1.0;
        for (int j = 0; j < n; j++)
        {
            if (d[j] < 0)
            {
                lambda = min(lambda, -0.9 / d[j]);
            }
                
        }
        //cout << "lambda: " << lambda << endl;
        z = e + lambda * d;
        xprev = x;
        x = D * z;
        
        /*cout << "z" << endl;
        for (int i = 0; i < n; i++)
            cout << z(i) << " " ;
        cout << "x" << endl;
        for (int i = 0; i < n; i++)
            cout << x(i) << " " ;
        cout << endl;*/
        cout << "Cost: " << c.transpose() * x << endl;
        err = x - xprev;
        cout << "Change: " << err.norm() << endl;
    }
    

    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}
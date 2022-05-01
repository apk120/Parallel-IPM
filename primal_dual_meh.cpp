// #include <bits/stdc++.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include "eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;

int main()
{
    clock_t tStart = clock();

    int m, n = 4;
    m = 2;
    MatrixXd A(m, n), X(n, n), S(n, n), At(n, m), D(n, n), ADA(m, m), AS(m, n);
    VectorXd x(n), y(m), s(n), c(n), b(m), dx_aff(n), dy_aff(m), ds_aff(n), dx_cor(n), dy_cor(m), ds_cor(n);
    VectorXd rd(n), rp(m), rc(n), v(n);
    double mu, sigma = 0.8, ap_aff, ad_aff, mu_aff, ap_k, ad_k;
    int iter = 10;

    //
    b(0) = 1;
    b(1) = 2;
    c(0) = -1, c(1) = -1, c(2) = 1, c(3) = 1;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A(i, j) = 0;
    A(0, 0) = 1, A(0, 2) = 1, A(1, 1) = 1, A(1, 3) = 1;
    cout << A << endl;
    x << 0.8, 0.1, 0.2, 1.9;
    s << 0.1, 0.2, 2.1, 2.2;
    At = A.transpose();
    y = At.colPivHouseholderQr().solve(c - s);
    cout << "y: ";
    for (int i = 0; i < m; i++)
        cout << y(i) << " ";
    cout << endl;
    cout << A.transpose() * y + s - c << endl;
    //
    for (int i = 0;  i < n; i++)
        for (int j = 0; j < n; j++)
        {
            X(i, j) = 0;
            S(i, j) = 0;
        }
    for (int i = 0; i < iter; i++)
    {
        for (int j = 0; j < n; j++)
        {
            X(j, j) = x(j);
            S(j, j) = s(j);
        }

        mu = (x.transpose() * s);
        mu = mu / n;
        cout << "mu :" << mu << endl;
        rd = c - s - A.transpose() * y;
        rp = b - A * x;
        for (int j = 0; j < n; j++)
            rc(j) = - x(j) * s(j); //0.65 * mu - x(j) * s(j)
        
        D = S.inverse() * X;
        ADA = A * (D * A.transpose());
        AS = A * S.inverse();
        dy_aff = -1.0 * (ADA.inverse() * (AS * rc - A * D * rd - rp));
        ds_aff = -1.0 * At * dy_aff + rd;
        dx_aff = S.inverse() * rc - D * ds_aff;

        ap_aff = 1.0;
        ad_aff = 1.0;

        /*cout << "dy_aff: " << endl;
        cout << AS*rc << endl;
        cout <<"d2: "<< endl;
        cout << A*D*rd << endl;
        cout << "d3: " << endl;
        cout << AS * rc - A * D * rd - rp << endl;*/
        for (int j = 0; j < n; j++)
        {
            if (dx_aff(j) < 0)
                ap_aff = min(ap_aff, -0.9*x(j)/dx_aff(j));
            if (ds_aff(j) < 0)
                ad_aff = min(ad_aff, -0.9*s(j)/ds_aff(j));
        } 
        
        // x = x +  ap_aff * dx_aff;
        // y = y + ad_aff * dy_aff;
        // s = s + ad_aff * ds_aff;
        // mu_aff = (x).transpose() * (s);
        // mu_aff = mu_aff / n;
        // cout << "mu _ aff : " << mu_aff << endl;
        /*Correct till here*/
        //
        
        mu_aff = (x + ap_aff * dx_aff).transpose() * (s + ad_aff * ds_aff);
        mu_aff = mu_aff / n;
        //if (i < 3)
          //  mu_aff = 0.99*mu;
        cout << "mu _ aff : " << mu_aff << endl;
        sigma = pow((mu_aff/mu), 3);
        cout << "sigma: " << sigma << endl;
        for (int j = 0; j < n; j++)
            v(j) = sigma * mu - dx_aff(j) * ds_aff(j);

        
        // dy_cor = -1.0 * (ADA.inverse() * (AS * v));
        // ds_cor = -1.0 * At * dy_cor;
        // dx_cor = S.inverse() * rc - D * ds_cor;
        ds_cor = X.inverse() * v;
        dy_cor = -1.0 * (At.colPivHouseholderQr().solve(ds_cor));
        //
        dy_cor = dy_aff + dy_cor;
        // dx_cor = dx_aff + dx_cor;
        dx_cor = dx_aff;
        ds_cor = ds_aff + ds_cor;
        // cout << "dx_cor before: " << endl;
        // cout << dx_cor << endl;
        ap_k = 1.0, ad_k = 1.0;
        for (int j = 0; j < n; j++)
        {
            if (dx_cor(j) < 0)
                ap_k = min(ap_k, -0.9*x(j)/dx_cor(j));
            if (ds_cor(j) < 0)
                ad_k = min(ad_k, -0.9*s(j)/ds_cor(j));
        } 

        x = x +  ap_k * dx_cor;
        y = y + ad_k * dy_cor;
        s = s + ad_k * ds_cor;


        cout << "x : " << endl;
        cout << x << endl;
        cout << "Cost: " << c.transpose() * x << endl;
        cout << "Change: " << dx_aff.norm() << endl;

    }
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}
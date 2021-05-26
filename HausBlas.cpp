#include <stdint.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cblas.h>
#include <omp.h>
using namespace std;

inline int8_t sign (double a)
{
  if (a > 0)
    return 1;
  else if (a < 0)
    return -1;
  else
    return 0;
}

//The QR factorization function
inline void QR_fact (double* Q, double* R, uint32_t n)
{
  double arr_norm = 0;
  double *arr = new double[n]; //processed array
  double* w = new double[n];
  double *aux_matrix = new double[n * n];
  uint32_t arr_sz = n;
  uint32_t hght = 0;

  //Body of the Algorithm
  for (uint32_t clmn = 0; clmn < n; clmn++)
  {
    hght = clmn * n;
    for (uint32_t h = 0; h < arr_sz; h++)
    {
      arr[h] = R[hght + clmn];
      hght+= n;
    }

    arr_norm = cblas_dnrm2(arr_sz, arr, 1);
    if (arr_norm != 0)
    {
      uint32_t aux = n - arr_sz;

      //creating w vector
      arr[0] += arr_norm * sign(arr[0]);
      arr_norm = cblas_dnrm2(arr_sz, arr, 1);
      for (uint32_t i = 0; i < aux; i++)
        w[i] = 0;
      for (uint32_t i = aux; i < n; i++)
        w[i] = arr[i - aux] / arr_norm;

      //Q [T] = Q_step * Q [T]
      cblas_dgemv(CblasRowMajor, CblasTrans, n, n, 1.0, Q, n, w, 1, 0, arr, 1);
      cblas_dger(CblasRowMajor, n, n, -2, w, 1, arr, 1, Q, n);

      //R = Q_step * R
      cblas_dgemv(CblasRowMajor, CblasTrans, n, n, 1.0, R, n, w, 1, 0, arr, 1 );
      cblas_dger(CblasRowMajor, n, n, -2, w, 1, arr, 1, R, n);

    }
    arr_sz--;
  }
  //Q [T] -> Q
  for (uint32_t h = 0; h < n; h++)
  {
    for (uint32_t wd = 0; wd < n; wd++)
      aux_matrix[wd + h * n] = Q[wd * n + h];
  }
  hght = 0;
  for (uint32_t h = 0; h < n; h++)
  {
    for (uint32_t wd = 0; wd < n; wd++)
      Q[wd + hght] = aux_matrix[hght + wd];
    hght += n;
  }
  delete [] w;
  delete [] aux_matrix;
  delete [] arr;
}

//There is a system of linear equations Ax = b; A is a matrix of n * n size
//This program implements The Householder QR-factorization method
int main()
{
  double start = omp_get_wtime();
  cout << "Starting non parallel Householder method implementation\n";
  
  //Entering equation data
  uint32_t n = 1024;
  const uint32_t ms = n * n;
  double *A = new double[ms];
  double *Q = new double[ms];
  double *R = new double[ms];
  double *M_aux = new double[ms];
  uint32_t height = 0;

  //Preliminary work for A, Q and R matrices
  height = 0;
  uint32_t diag = 0;
  for (uint32_t h = 0; h < n; h++)
  {
    for (uint32_t w = 0; w < n; w++)
    {
      uint32_t coord = w + height;
      A[coord] =  1 / double(w + h + 2);
      R[coord] = A[coord];
      Q[coord] = 0;
    }
    Q[height + diag] = 1;
    height += n;
    diag++;
  }

  QR_fact(Q, R, n);
  
  double end = omp_get_wtime();
  cout << "time:" << (end - start) << "\n";


  //Computing ||QR - A|| / ||A||
  for (uint32_t h = 0; h < ms; h++)
    M_aux[h] = A[h];
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Q, n, R,
    n, -1, M_aux, n);



  double err = cblas_dnrm2(ms, M_aux, 1);
  err /= cblas_dnrm2(n, A, 1);
  cout << "accuracy: " << err << "\n";

}

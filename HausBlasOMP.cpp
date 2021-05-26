#include <stdint.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <stdio.h>

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
  uint32_t threads = omp_get_num_procs();
  cout << "total threads:" << threads << "\n";
  double arr_norm = 0;
  
  double *arr = new double[n]; //processed array
  double* w = new double[n];
  double* aux_matrix = new double[n * n];


  uint32_t size[threads];
  for (uint16_t a = 0; a < threads; a++)
  {
    size[a] = n / threads + (a < (n % threads));
  }
     
  #pragma omp parallel num_threads(threads)
  {
    uint32_t arr_sz = n;
    uint32_t hght = 0;
    int my_rank = omp_get_thread_num();
    uint32_t shift = 0;
    for (uint16_t a = 0; a < my_rank; a++)
    {
      shift += size[a];
    }  
    double* Q_start = Q + shift * n;
    double* R_start = R + shift * n;
    double* R_colstart = R + shift;
    double* Q_colstart = Q + shift;
    double* w_start = w + shift;
    double* arr_start = arr + shift;
  
      
    //Body of the Algorithm
    for (uint32_t clmn = 0; clmn < n; clmn++)
    {
      if (my_rank == 0)
      {  
        hght = clmn * n;
        for (uint32_t h = 0; h < arr_sz; h++)
        {
          arr[h] = R[hght + clmn];
          hght+= n;
        }        
        arr_norm = cblas_dnrm2(arr_sz, arr, 1);
      }

      #pragma omp barrier
      if (arr_norm != 0)
      {
        uint32_t aux = n - arr_sz;
        //creating w vector
          
        if (my_rank == 0)
        {
          arr[0] += arr_norm * sign(arr[0]);
          arr_norm = cblas_dnrm2(arr_sz, arr, 1);
           
          for (uint32_t i = 0; i < aux; i++)
            w[i] = 0;
          for (uint32_t i = aux; i < n; i++)
            w[i] = arr[i - aux] / arr_norm;
        }        
        #pragma omp barrier
        
        cblas_dgemv(CblasRowMajor, CblasTrans, n, size[my_rank], 1.0, Q_colstart,
          n, w, 1, 0, arr_start, 1);
        #pragma omp barrier
                        
        cblas_dger(CblasRowMajor, size[my_rank], n, -2, w_start, 1, arr, 1,
          Q_start, n);
        #pragma omp barrier   

        
        cblas_dgemv(CblasRowMajor, CblasTrans, n, size[my_rank], 1.0, R_colstart,
          n, w, 1, 0, arr_start, 1);
        #pragma omp barrier   
                
        cblas_dger(CblasRowMajor, size[my_rank], n, -2, w_start, 1, arr, 1,
          R_start, n);
        #pragma omp barrier
        
      }
      arr_sz--;
    }
  }
  //Q [T] -> Q
  for (uint32_t h = 0; h < n; h++)
  {
    for (uint32_t wd = 0; wd < n; wd++)
      aux_matrix[wd + h * n] = Q[wd * n + h];
  }
  uint32_t hght = 0;
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
  cout << "Starting OMP parallel Householder method implementation\n";
  double start = omp_get_wtime();
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
  cout << "Qr finished\n";
  double end = omp_get_wtime();
  cout << "time: " << (end - start) << "\n";
  





  //Computing ||QR - A|| / ||A||
  for (uint32_t h = 0; h < ms; h++)
    M_aux[h] = A[h];
  uint32_t threads = omp_get_num_procs();
  uint16_t blocks = sqrt(threads);
  uint16_t total_blocks = blocks * blocks;
  uint32_t size[blocks];
  for (uint16_t a = 0; a < blocks; a++)
  {
    size[a] = n / blocks + (a < (n % blocks));
  }
  
  #pragma omp parallel num_threads(total_blocks)
  {
    int my_block = omp_get_thread_num();
    uint16_t block_vert = my_block / blocks;
    uint16_t block_hor = my_block % blocks;
    uint32_t vert_shift = 0;
    uint32_t hor_shift = 0;
    for (uint32_t a = 0; a < block_hor; a++)
    {
      hor_shift += size[a]; 
    }
    for (uint32_t a = 0; a < block_vert; a++)
    {
      vert_shift += size[a]; 
    }
    vert_shift *= n;
    double* Q_start = Q + vert_shift;
    double* R_start = R + hor_shift;
    double* M_aux_start = M_aux + hor_shift + vert_shift;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size[block_vert],
      size[block_hor], n, 1, Q_start, n, R_start, n, -1, M_aux_start, n);

  }


  
  double err = cblas_dnrm2(ms, M_aux, 1);
  err /= cblas_dnrm2(n, A, 1);
  cout << "accuracy: " << err << "\n";


  cout << "wtime omp\n"; 
}

#include <stdint.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <stdio.h>
#include <mpi.h>

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
  


//There is a system of linear equations Ax = b; A is a matrix of n * n size
//This program implements The Householder QR-factorization method
int main(int argc, char ** argv)
{
 
  MPI_Init(&argc, &argv);
  cout << "Starting MPI parallel Householder method implementation\n";
 
  double start = MPI_Wtime(); 
  int total_pcs = 0;
  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_pcs);
  MPI_Request ierr[1];  
  
  unsigned long int n = 1024;
  unsigned long int columns = n / total_pcs;
  if (my_rank == (total_pcs - 1))
    columns += n % total_pcs;
  unsigned long int ms = n * columns;
  unsigned long int big_ms = n * n;
  unsigned long int small_ms = columns * columns;
  double *A = new double[ms];
  double *Q = new double[ms]; 
  double *R = new double[ms];
  double *R_aux = new double[ms];
  double *A_aux = new double[ms];
  double *M = new double[small_ms];

  
  
  //filling A and R matrices 
  uint32_t height = 0;
  height = 0;
  for (uint32_t h = 0; h < n; h++)
  {
    for (uint32_t w = 0; w < columns; w++)
    {
      uint32_t coord = w + height;
      A[coord] = 1 / double(w + h + 2) + my_rank;
      R[coord] = A[coord];
    }
    height += columns;
  }
  
  //filling Q matrix
  unsigned long int message[1];
  unsigned long int columns_before = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  total_pcs--;
  for (uint16_t i = 0; i < total_pcs; i++)
  {
    if (my_rank == i)
    {
      message[0] = columns_before + columns;
      MPI_Send(message, 1, MPI_UNSIGNED_LONG, i + 1, 0, MPI_COMM_WORLD); 
    }
    if (my_rank == i + 1)
    {
      MPI_Recv(message, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      columns_before = message[0];
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  total_pcs++;
  

  for (uint32_t i = 0; i < ms; i++)
  {
    Q[i] = 0;
  }
  for (uint32_t i = 0; i < columns; i++)
  {
    Q[(columns_before + i) * columns + i] = 1;
    
  }

  double norm[1] = {0};
  double *segment_arr = new double[columns];
  double *storage = new double[n];
  double *w = new double[n];
  long unsigned int processed_columns[1] = {columns};
  
  uint16_t segment_nmb = 0;
  uint32_t step = 0;
  uint32_t storage_sz = n;
  uint32_t hght = 0;
  
  //sending number of columns from segment_nmb == 0 to other processes
  if (my_rank == 0)
  {
    for (int i = 1; i < total_pcs; i++)
    {
      MPI_Isend(processed_columns, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, ierr);
    }
  }
  if (my_rank != 0)
    MPI_Recv(processed_columns, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  


   
  cout << "got to QR-factrorization\n";    
  //Body of the QR-factorization
  for (uint32_t index = 0; index < n; index++)
  {
    //cout << "enter index" << index << "\n";
    if (step >= processed_columns[0])
    {
      step = 0;
      segment_nmb++;
      if (my_rank == segment_nmb)
      {
        processed_columns[0] = columns;
        for (int i = 0; i < segment_nmb; i++)
        {
          MPI_Isend(processed_columns, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, ierr);
        }
        for (int i = segment_nmb + 1; i < total_pcs; i++)
        {
          MPI_Isend(processed_columns, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, ierr);
        }
      }
      if (my_rank != segment_nmb)
        MPI_Recv(processed_columns, 1, MPI_UNSIGNED_LONG, segment_nmb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    if (my_rank == segment_nmb)
    {
      hght = index * columns;
      //cout << "\n" << step << "\n ";
      for (uint32_t h = 0; h < storage_sz; h++)
      {
        storage[h] = R[hght + step];
        hght+= columns;
      }        
      norm[0] = cblas_dnrm2(storage_sz, storage, 1);
      
      
      for (int i = 0; i < segment_nmb; i++)
      {
        MPI_Isend(norm, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, ierr);
        MPI_Isend(storage, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, ierr);        
      }
      for (int i = segment_nmb + 1; i < total_pcs; i++)
      {
        MPI_Isend(norm, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, ierr);
        MPI_Isend(storage, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, ierr);        
      }
    }
    if (my_rank != segment_nmb)
    {
      MPI_Recv(norm, 1, MPI_DOUBLE, segment_nmb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(storage, n, MPI_DOUBLE, segment_nmb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);   
    //passing storage && norm to other processes
    if (norm[0] != 0)
    {
      //creating w vector using storage
      uint32_t aux = n - storage_sz;              
      storage[0] += norm[0] * sign(storage[0]);
      norm[0] = cblas_dnrm2(storage_sz, storage, 1);
      for (uint32_t i = 0; i < aux; i++)
        w[i] = 0;
      for (uint32_t i = aux; i < n; i++)
        w[i] = storage[i - aux] / norm[0];
      
      cblas_dgemv(CblasRowMajor, CblasTrans, n, columns, 1.0, Q, columns, w, 1, 0,
        segment_arr, 1);                               
      cblas_dger(CblasRowMajor, n, columns, -2, w, 1, segment_arr, 1, Q, columns);
      
      cblas_dgemv(CblasRowMajor, CblasTrans, n, columns, 1.0, R, columns, w, 1, 0,
        segment_arr, 1);                               
      cblas_dger(CblasRowMajor, n, columns, -2, w, 1, segment_arr, 1, R, columns);

    }
    MPI_Barrier(MPI_COMM_WORLD);
    step++;
    storage_sz--;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double finish = MPI_Wtime();

  
  //Computing ||QR - A|| / ||A||
  cout << "Computing ||QR - A|| / ||A||\n";
  if (n % processed_columns[0] > 0)
  {
    cout << "n should be equal to integer * process amount/n";
    exit(0);
  }
  int A_tag = total_pcs;
  int R_tag = total_pcs + 1;

  
  for (uint8_t i = 0; i < total_pcs; i++)
  {
    MPI_Isend(R, ms, MPI_DOUBLE, i, R_tag, MPI_COMM_WORLD, ierr);
   
    MPI_Isend(A, ms, MPI_DOUBLE, i, A_tag, MPI_COMM_WORLD, ierr);
  }

  
  for (uint8_t i = 0; i < total_pcs; i++)
  {
    MPI_Recv(R_aux, ms, MPI_DOUBLE, i, R_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(A_aux, ms, MPI_DOUBLE, i, A_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    unsigned long int coord = my_rank * small_ms;    
    for (unsigned long int c = 0; c < small_ms; c++)
    {
      M[c] = A_aux[coord + c];
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, columns, columns, n, 1.0,
      Q, columns, R_aux, columns, -1, M, columns);
    
    MPI_Isend(M, small_ms, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, ierr);
          
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Isend(A, ms, MPI_DOUBLE, 0, A_tag, MPI_COMM_WORLD, ierr);  

  
  cout << "got to computing norm \n";
  if (my_rank == 0)
  {
    double *M_norm = new double[big_ms];
    double *A_norm = new double[big_ms];
    for (int i = 0; i < total_pcs; i++)
    {
      MPI_Recv(A, ms, MPI_DOUBLE, i, A_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      unsigned long int A_coord = i * ms;
      for (unsigned long int j = 0; j < ms; j++)
      {
        A_norm[A_coord + j] = A[j];
      }
      for (int tag = 0; tag < total_pcs; tag++)
      {
        MPI_Recv(M, small_ms, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        unsigned long int coord = i * ms + tag * small_ms;
        for (unsigned long int s = 0; s < small_ms; s++)
        {
          M_norm[coord + s] = M[s];
        }
      }
    }
        
    double err = cblas_dnrm2(big_ms, M_norm, 1);
    err /= cblas_dnrm2(big_ms, A_norm, 1);
    cout << err << "\n";
  }
  
  
  MPI_Finalize();
  cout << (finish - start) << "\n";
}

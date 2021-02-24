#include <stdio.h>
#include "mpi.h"
#include <iostream>
using namespace std;
#include <cmath>
#include <fstream>
#include <iomanip>

int main(){
    int M = 2305; 
    float T = 1, X = 2;

    float dt = 0.2/M, dx = X/M;
    float x, y;
    int N = T/dt;

    float tstart, tend;
    float runtime;

    //initialize MPI    
    int nproc, rank;
    MPI_Status status;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = (M+1-2)/nproc + 2;
    float U[size][M+1], U_temp1[size][M+1], U_temp2[size][M+1]; // Initialize two vectors, U and U_temp, U for store results, U_temp for updating.

    
    tstart = MPI_Wtime();
    // Give initial value to U.
    for(int i = 0; i < nproc ; i++){
        if(rank==i){   
            for(int m = 0; m < size ; m++){     
                for(int n = 1; n < M ; n++){                  
                    x = (i*(size-2)+m)*dx - 1;
                    y = n*dx - 1;
                    U[m][n] = exp( -40 * ((x-0.4)*(x-0.4) + y*y) ); 
                    U_temp1[m][n] = U[m][n]; 
                }                        
            }
            for(int m = 0; m<size ; m++){
                U[m][0] = 0;
                U[m][M] = 0;
                U_temp1[m][0] = 0;
                U_temp1[m][M] = 0;
                U_temp2[m][0] = 0;
                U_temp2[m][M] = 0;
            }                
        }
    }
    // Set boundary points to be 0
    if(rank==0){
        for(int n = 1; n<M ; n++){
            U[0][n] = 0;
            U_temp1[0][n] = 0;
            U_temp2[0][n] = 0;
        }
    }
    if(rank==nproc-1){
        for(int n = 1; n<M ; n++){
            U[size-1][n] = 0;
            U_temp1[size-1][n] = 0;
            U_temp2[size-1][n] = 0;
        }
    }

    // Update value of U.
    for(int j = 0; j < N; j++){
        for(int i = 0; i<nproc ; i++){
            if(rank == i){
                for(int m = 1; m<size-1 ; m++){
                    for(int n = 1; n<M ; n++){
                        U_temp2[m][n] = (dt*dt)/(dx*dx) * (U_temp1[m-1][n] + U_temp1[m+1][n] + U_temp1[m][n+1] + U_temp1[m][n-1] - 4*U_temp1[m][n]) + 2*U_temp1[m][n] - U[m][n]; 
                    }
                    
                }
                if(rank != 0){ // Except rank 0, send first value to current_rank-1  and receive value from current_rank-1
                    for(int n = 0; n<M ; n++){
                        MPI_Send(&U_temp2[1][n],1,MPI_FLOAT,(nproc+i-1)%nproc, i+10*n, MPI_COMM_WORLD); 
                        MPI_Recv(&U_temp2[0][n],1,MPI_FLOAT,(nproc+i-1)%nproc, (nproc+i-1)%nproc+10*n, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);                        
                    }

                }
                if(rank != nproc-1){// Except last rank, send first value to current_rank+1  and receive value from current_rank+1
                    for(int n = 0; n<M ; n++){
                        MPI_Send(&U_temp2[size-2][n],1,MPI_FLOAT,(nproc+i+1)%nproc, i+10*n, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[size-1][n],1,MPI_FLOAT,(nproc+i+1)%nproc, (nproc+i+1)%nproc+10*n, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    }
                }                

                MPI_Barrier(MPI_COMM_WORLD); // wait all process finish sending and receiving, update U.
                for(int m = 0; m<size ; m++){
                    for(int n = 0; n<M+1; n++){
                        U[m][n] = U_temp1[m][n];
                        U_temp1[m][n] = U_temp2[m][n]; 
                        U_temp2[m][n] = 0;
                    }
                }
            }
        } 
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    tend = MPI_Wtime();
    runtime = tend - tstart;       
    cout<<"t: "<<to_string(runtime) << endl;


    MPI_Finalize();

    return 0;


}
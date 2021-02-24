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
    float U[M+1][size], U_temp1[M+1][size], U_temp2[M+1][size]; // Initialize three vectors, U and U_temp, U for store results, U_temp for updating.
    
    tstart = MPI_Wtime();
    // Give initial value to U.
    for(int i = 0; i < nproc ; i++){
        if(rank==i){   
            for(int m = 0 ; m < M+1 ; m++){     
                for(int n = 0; n < size ; n++){                  
                    x = m*dx - 1;
                    y = (i*(size-2)+n)*dx - 1;
                    U[m][n] = exp( -40 * ((x-0.4)*(x-0.4) + y*y) ); 
                    U_temp1[m][n] = U[m][n];    
                }                        
            }
            for(int n = 0; n<size ; n++){
                U[0][n] = 0;
                U[M][n] = 0;
                U_temp1[0][n] = 0;
                U_temp1[M][n] = 0;
                U_temp2[0][n] = 0;
                U_temp2[M][n] = 0;
            }                
        }
    }
    // Set boundary points to be 0
    if(rank==0){
        for(int m = 1; m<M ; m++){
            U[m][0] = 0;
            U_temp1[m][0] = 0;
            U_temp2[m][0] = 0;
        }
    }
    if(rank==nproc-1){
        for(int m = 1; m<M ; m++){
            U[m][size-1] = 0;
            U_temp1[m][size-1] = 0;
            U_temp2[m][0] = 0;
        }
    }


    // Update value of U.
    for(int j = 0; j < N; j++){
        for(int i = 0; i<nproc ; i++){
            if(rank == i){
                for(int m = 1; m<M ; m++){
                    for(int n = 1; n<size-1 ; n++){
                        U_temp2[m][n] = (dt*dt)/(dx*dx) * (U_temp1[m-1][n] + U_temp1[m+1][n] + U_temp1[m][n+1] + U_temp1[m][n-1] - 4*U_temp1[m][n]) + 2*U_temp1[m][n] - U[m][n]; 
                    }
                    
                }
                if(rank != 0){ // Except rank 0, send first value to current_rank-1  and receive value from current_rank-1
                    for(int m = 0; m<M ; m++){
                        MPI_Send(&U_temp2[m][1],1,MPI_FLOAT,(nproc+i-1)%nproc, i+10*m, MPI_COMM_WORLD); 
                        MPI_Recv(&U_temp2[m][0],1,MPI_FLOAT,(nproc+i-1)%nproc, (nproc+i-1)%nproc+10*m, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);                        
                    }

                }
                if(rank != nproc-1){// Except last rank, send first value to current_rank+1  and receive value from current_rank+1
                    for(int m = 0; m<M ; m++){
                        MPI_Send(&U_temp2[m][size-2],1,MPI_FLOAT,(nproc+i+1)%nproc, i+10*m, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[m][size-1],1,MPI_FLOAT,(nproc+i+1)%nproc, (nproc+i+1)%nproc+10*m, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    }
                }                

                MPI_Barrier(MPI_COMM_WORLD); // wait all process finish sending and receiving, update U.
                for(int m = 0; m<M+1 ; m++){
                    for(int n = 0; n<size; n++){
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
    if(rank==0){        
        cout<<"t: "<<runtime << endl;
    }

    MPI_Finalize();

    return 0;


}
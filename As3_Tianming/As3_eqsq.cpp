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
    int a, b;
    int N = T/dt;

    float tstart, tend;
    float runtime;

    //initialize MPI    
    int nproc, rank;
    
    MPI_Status status;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int sq_nproc = sqrt(nproc);

    int size = (M+1-2)/sq_nproc + 2; 
    float U[size][size], U_temp1[size][size], U_temp2[size][size]={0}; // Initialize two vectors, U and U_temp, U for store results, U_temp for updating.
    // Initialize U_result to store final results.

    tstart = MPI_Wtime();
    // Give initial value to U.
    for(int i = 0; i < nproc ; i++){
        if(rank==i){   
            for(int m = 0; m < size ; m++){     
                for(int n = 0; n < size ; n++){    
                    a = i/int(sqrt(nproc));
                    b = i%sq_nproc;             
                    x = (a*(size-2)+m)*dx - 1;
                    y = (b*(size-2)+n)*dx - 1;
                    U[m][n] = exp( -40 * ((x-0.4)*(x-0.4) + y*y) ); 
                    U_temp1[m][n] = U[m][n];    
                }                        
            }
            // Set boundary points to be 0
            if(rank%sq_nproc==0){
                for(int m = 0; m<size ; m++){
                    U[m][0] = 0;
                    U_temp1[m][0] = 0;
                    U_temp2[m][0] = 0;                       
                }
            }
            if(rank%sq_nproc==sq_nproc-1){

                for(int m = 0; m<size ; m++){
                    U[m][size-1] = 0;
                    U_temp1[m][size-1] = 0;
                    U_temp2[m][size-1] = 0;                       
                }
            }
            if(rank/sq_nproc==0){
                for(int n = 0; n<size ; n++){
                    U[0][n] = 0;
                    U_temp1[0][n] = 0;  
                    U_temp2[0][n] = 0;                     
                }
            }
            if(rank/sq_nproc==sq_nproc-1){
                for(int n = 0; n<size ; n++){
                    U[size-1][n] = 0;
                    U_temp1[size-1][n] = 0; 
                    U_temp2[size-1][n] = 0;                      
                }
            }
     
        }
    }


    // Update value of U.
    for(int j = 0; j < N; j++){
        for(int i = 0; i<nproc ; i++){
            if(rank == i){
                for(int m = 1; m<size-1 ; m++){
                    for(int n = 1; n<size-1 ; n++){
                        U_temp2[m][n] = (dt*dt)/(dx*dx) * (U_temp1[m-1][n] + U_temp1[m+1][n] + U_temp1[m][n+1] + U_temp1[m][n-1] - 4*U_temp1[m][n]) + 2*U_temp1[m][n] - U[m][n]; 
                    }
                    
                }
                if(rank/sq_nproc != 0){ // Except rank 0, send first value to current_rank-1  and receive value from current_rank-1
                    for(int n = 1; n<size-1 ; n++){
                        MPI_Send(&U_temp2[1][n],1,MPI_FLOAT,i-sq_nproc, i+10*n, MPI_COMM_WORLD); 
                        MPI_Recv(&U_temp2[0][n],1,MPI_FLOAT,i-sq_nproc, (i-sq_nproc)+10*n, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);                        
                    }

                }
                if(rank/sq_nproc != sq_nproc-1){// Except last rank, send first value to current_rank+1  and receive value from current_rank+1
                    for(int n = 1; n<size-1 ; n++){
                        MPI_Send(&U_temp2[size-2][n],1,MPI_FLOAT,i+sq_nproc, i+10*n, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[size-1][n],1,MPI_FLOAT,i+sq_nproc, (i+sq_nproc)+10*n, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    }
                }  
                if(rank%sq_nproc != 0){// Except last rank, send first value to current_rank+1  and receive value from current_rank+1
                    for(int m = 1; m<size-1 ; m++){
                        MPI_Send(&U_temp2[m][1],1,MPI_FLOAT,i-1, i+10*m, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[m][0],1,MPI_FLOAT,i-1, (i-1)+10*m, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    }
                }  
                if(rank%sq_nproc != sq_nproc-1){// Except last rank, send first value to current_rank+1  and receive value from current_rank+1
                    for(int m = 1; m<size-1 ; m++){
                        MPI_Send(&U_temp2[m][size-2],1,MPI_FLOAT,i+1, i+10*m, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[m][size-1],1,MPI_FLOAT,i+1, (i+1)+10*m, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                    }
                } 
                // Swap corner points
                if(rank/sq_nproc != 0  && rank%sq_nproc != 0){// Upper left
                        MPI_Send(&U_temp2[1][1],1,MPI_FLOAT,i-sq_nproc-1, 1, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[0][0],1,MPI_FLOAT,i-sq_nproc-1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                } 
                if(rank/sq_nproc != sq_nproc-1  && rank%sq_nproc != sq_nproc-1){// Lower right
                        MPI_Send(&U_temp2[size-2][size-2],1,MPI_FLOAT,i+sq_nproc+1, 1, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[size-1][size-1],1,MPI_FLOAT,i+sq_nproc+1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                } 
                if(rank/sq_nproc != 0  && rank%sq_nproc != sq_nproc-1){// Upper right
                        MPI_Send(&U_temp2[1][size-2],1,MPI_FLOAT,i-sq_nproc+1, 1, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[0][size-1],1,MPI_FLOAT,i-sq_nproc+1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                } 
                if(rank/sq_nproc != sq_nproc-1  && rank%sq_nproc != 0){// Lower left
                        MPI_Send(&U_temp2[size-2][1],1,MPI_FLOAT,i+sq_nproc-1, 1, MPI_COMM_WORLD);
                        MPI_Recv(&U_temp2[size-1][0],1,MPI_FLOAT,i+sq_nproc-1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
                } 


                MPI_Barrier(MPI_COMM_WORLD); // wait all process finish sending and receiving, update U.
                for(int m = 0; m<size ; m++){
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



    /*for(int k = 0; k<nproc; k++){
        if (rank == k){          
            fstream myfile;
            string name("result_eqsq");
            string txt(".txt");
            name = name + to_string(k) + txt;
            
            myfile.open(name,fstream::out);
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    myfile << U_temp1[i][j] << ",";
                }       
                myfile << endl; 
            }    
            myfile.close();
        } 
    }*/
    return 0;
}
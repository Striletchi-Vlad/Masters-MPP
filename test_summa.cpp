#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "mpi.h"

void readInput(int rows, int cols, float *data) {
    // For testing purposes: create a checkerboard pattern
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i * cols + j] = (i + j) % 2;
}

void printOutput(int rows, int cols, float *data) {
    FILE *fp = fopen("outSUMMA.txt", "wb");
    if (fp == NULL) {
        std::cout << "ERROR: Output file outSUMMA.txt could not be opened" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            fprintf(fp, "%lf ", data[i * cols + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main (int argc, char *argv[]){
    MPI_Init(&argc, &argv);

    // Get total processes and rank from MPI_COMM_WORLD
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Compute grid dimension and number of active processes (largest perfect square)
    int gridDim = (int) sqrt(worldSize);
    int activeProcs = gridDim * gridDim;

    // Split communicator: processes with rank < activeProcs are active, others idle.
    MPI_Comm activeComm;
    int color = (worldRank < activeProcs) ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &activeComm);
    if (activeComm == MPI_COMM_NULL) {
        // Idle process: simply exit.
        MPI_Finalize();
        return 0;
    }

    // Now use activeComm for the computation.
    int activeRank, activeSize;
    MPI_Comm_rank(activeComm, &activeRank);
    MPI_Comm_size(activeComm, &activeSize);
    // activeSize should equal activeProcs

    if(argc < 4){
        if(activeRank == 0){
            std::cout << "ERROR: The syntax of the program is ./summa m k n" << std::endl;
        }
        MPI_Abort(activeComm, 1);
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    // Instead of requiring the number of processes to be square, we now require
    // that m, n, and k be divisible by gridDim (derived from activeProcs)
    if((m % gridDim) || (n % gridDim) || (k % gridDim)){
        if(activeRank == 0)
            std::cout << "ERROR: 'm', 'k' and 'n' must be multiples of " << gridDim << std::endl;
        MPI_Abort(activeComm, 1);
    }

    if((m < 1) || (n < 1) || (k < 1)){
        if(activeRank == 0)
            std::cout << "ERROR: 'm', 'k' and 'n' must be higher than 0" << std::endl;
        MPI_Abort(activeComm, 1);
    }

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    if(activeRank == 0){
        A = new float[m * k];
        readInput(m, k, A);
        B = new float[k * n];
        readInput(k, n, B);
        C = new float[m * n];
    }

    // Divide matrices into blocks
    int blockRowsA = m / gridDim;
    int blockRowsB = k / gridDim;
    int blockColsB = n / gridDim;

    // Create MPI datatypes for the blocks
    MPI_Datatype blockAType, blockBType, blockCType;
    MPI_Type_vector(blockRowsA, blockRowsB, k, MPI_FLOAT, &blockAType);
    MPI_Type_commit(&blockAType);
    MPI_Type_vector(blockRowsB, blockColsB, n, MPI_FLOAT, &blockBType);
    MPI_Type_commit(&blockBType);
    MPI_Type_vector(blockRowsA, blockColsB, n, MPI_FLOAT, &blockCType);
    MPI_Type_commit(&blockCType);

    float* myA = new float[blockRowsA * blockRowsB];
    float* myB = new float[blockRowsB * blockColsB];
    float* myC = new float[blockRowsA * blockColsB]();  // initialize to 0
    float* buffA = new float[blockRowsA * blockRowsB];
    float* buffB = new float[blockRowsB * blockColsB];

    MPI_Barrier(activeComm);
    double start = MPI_Wtime();

    MPI_Request req;
    // Scatter A and B from process 0 to all active processes using activeComm.
    if(activeRank == 0){
        for (int i = 0; i < gridDim; i++){
            for (int j = 0; j < gridDim; j++){
                int destRank = i * gridDim + j;
                MPI_Isend(A + i * blockRowsA * k + j * blockRowsB, 1, blockAType,
                          destRank, 0, activeComm, &req);
                MPI_Isend(B + i * blockRowsB * n + j * blockColsB, 1, blockBType,
                          destRank, 0, activeComm, &req);
            }
        }
    }

    MPI_Recv(myA, blockRowsA * blockRowsB, MPI_FLOAT, 0, 0, activeComm, MPI_STATUS_IGNORE);
    MPI_Recv(myB, blockRowsB * blockColsB, MPI_FLOAT, 0, 0, activeComm, MPI_STATUS_IGNORE);

    // Create communicators for rows and columns of the process grid using activeComm
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(activeComm, activeRank / gridDim, activeRank % gridDim, &rowComm);
    MPI_Comm_split(activeComm, activeRank % gridDim, activeRank / gridDim, &colComm);

    // Main loop of the SUMMA algorithm
    for (int p = 0; p < gridDim; p++){
        if (activeRank % gridDim == p){
            memcpy(buffA, myA, blockRowsA * blockRowsB * sizeof(float));
        }
        if (activeRank / gridDim == p){
            memcpy(buffB, myB, blockRowsB * blockColsB * sizeof(float));
        }
        MPI_Bcast(buffA, blockRowsA * blockRowsB, MPI_FLOAT, p, rowComm);
        MPI_Bcast(buffB, blockRowsB * blockColsB, MPI_FLOAT, p, colComm);

        // Multiply the submatrices and accumulate into myC
        for (int i_local = 0; i_local < blockRowsA; i_local++){
            for (int j_local = 0; j_local < blockColsB; j_local++){
                for (int l = 0; l < blockRowsB; l++){
                    myC[i_local * blockColsB + j_local] += 
                        buffA[i_local * blockRowsB + l] * buffB[l * blockColsB + j_local];
                }
            }
        }
    }

    // Gather the final matrix C on process 0
    if(activeRank == 0){
        // Copy local block from process 0
        for (int i = 0; i < blockRowsA; i++)
            memcpy(&C[i * n], &myC[i * blockColsB], blockColsB * sizeof(float));

        // Receive blocks from other processes
        for (int i = 0; i < gridDim; i++){
            for (int j = 0; j < gridDim; j++){
                if(i != 0 || j != 0){
                    MPI_Recv(&C[i * blockRowsA * n + j * blockColsB], 1, blockCType,
                             i * gridDim + j, 0, activeComm, MPI_STATUS_IGNORE);
                }
            }
        }
    } else {
        MPI_Send(myC, blockRowsA * blockColsB, MPI_FLOAT, 0, 0, activeComm);
    }

    double end = MPI_Wtime();

    if(activeRank == 0){
        std::cout << "Time with " << activeProcs << " processes: " << end - start << " seconds" << std::endl;
        printOutput(m, n, C);
        delete [] A;
        delete [] B;
        delete [] C;
    }

    MPI_Barrier(activeComm);

    delete [] myA;
    delete [] myB;
    delete [] myC;
    delete [] buffA;
    delete [] buffB;

    // Free the MPI datatypes and communicators
    MPI_Type_free(&blockAType);
    MPI_Type_free(&blockBType);
    MPI_Type_free(&blockCType);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&activeComm);

    MPI_Finalize();
    return 0;
}

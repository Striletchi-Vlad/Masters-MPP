#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <cmath>

// Helper: check if a file exists
bool fileExists(const std::string &filename) {
    std::ifstream f(filename);
    return f.good();
}

// Generate a random MxM matrix and save it in binary format.
void generateMatrix(const std::string &filename, int M) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Cannot open file " + filename + " for writing.");
    }
    std::vector<double> matrix(M * M);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < M * M; i++) {
        matrix[i] = dis(gen);
    }
    fout.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(double));
    fout.close();
}

// Read an MxM matrix sequentially from a binary file.
void sequentialReadMatrix(const std::string &filename, std::vector<double> &matrix, int M) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open file " + filename + " for reading.");
    }
    matrix.resize(M * M);
    fin.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));
    fin.close();
}

// Read an MxM matrix in parallel from a binary file.
void parallelReadMatrix(const std::string &filename, std::vector<double> &matrix, int M, int numThreads) {
    // Resize matrix to hold M*M elements.
    matrix.resize(M * M);
    
    // Total number of elements (doubles) to read.
    size_t totalDoubles = M * M;
    
    // Determine the base block size per thread and the remainder.
    size_t blockSize = totalDoubles / numThreads;
    size_t remainder = totalDoubles % numThreads;
    
    // Vector to hold the threads.
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    
    // Launch each thread to read its block.
    for (int i = 0; i < numThreads; i++) {
        // Calculate the start index and number of elements for this thread.
        size_t startIndex = i * blockSize;
        size_t count = blockSize;
        // The last thread gets any remaining elements.
        if (i == numThreads - 1) {
            count += remainder;
        }
        
        // Calculate the offset (in bytes) and total bytes to read.
        size_t byteOffset = startIndex * sizeof(double);
        size_t bytesToRead = count * sizeof(double);
        
        // Launch the thread.
        threads.emplace_back([&, byteOffset, bytesToRead, startIndex]() {
            // Each thread opens its own stream.
            std::ifstream fin(filename, std::ios::binary);
            if (!fin) {
                throw std::runtime_error("Cannot open file " + filename + " for reading.");
            }
            // Seek to the correct offset.
            fin.seekg(byteOffset);
            // Read the assigned block directly into the vector.
            fin.read(reinterpret_cast<char*>(matrix.data() + startIndex), bytesToRead);
            // No need to explicitly close the stream as it will be closed when fin goes out of scope.
        });
    }
    
    // Wait for all threads to complete.
    for (auto &t : threads) {
        t.join();
    }
}

// Write an MxM matrix to a text file (plaintext) for easy comparison.
void writeMatrix(const std::string &filename, const std::vector<double> &matrix, int M) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open file " + filename + " for writing.");
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            fout << matrix[i * M + j];
            if (j < M - 1) fout << " ";
        }
        fout << "\n";
    }
    fout.close();
}

// CUDA kernel for computing one element of the product matrix.
__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < M) {
        double sum = 0.0;
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

// Matrix multiplication using CUDA
void cudaMultiply(const std::vector<double> &A, const std::vector<double> &B,
                        std::vector<double> &C, int M, int numThreads, int blockSize) {
    int a_rows = A.size() / M;
    int a_cols = M;
    int b_rows = M;
    int b_cols = M;
    int c_rows = a_rows;
    int c_cols = b_cols;

    size_t size_a = a_rows * a_cols * sizeof(double);
    size_t size_b = b_rows * b_cols * sizeof(double);
    size_t size_c = c_rows * c_cols * sizeof(double);

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    
    // Allocate device memory.
    cudaError_t err = cudaMalloc((void**)&d_A, size_a);
    if (err != cudaSuccess) {
         std::cerr << "Error allocating memory for d_A: " << cudaGetErrorString(err) << std::endl;
         return;
    }
    err = cudaMalloc((void**)&d_B, size_b);
    if (err != cudaSuccess) {
         std::cerr << "Error allocating memory for d_B: " << cudaGetErrorString(err) << std::endl;
         cudaFree(d_A);
         return;
    }
    err = cudaMalloc((void**)&d_C, size_c);
    if (err != cudaSuccess) {
         std::cerr << "Error allocating memory for d_C: " << cudaGetErrorString(err) << std::endl;
         cudaFree(d_A);
         cudaFree(d_B);
         return;
    }
    
    // Copy host matrices to device memory.
    err = cudaMemcpy(d_A, A.data(), size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         std::cerr << "Error copying A to device: " << cudaGetErrorString(err) << std::endl;
         cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
         return;
    }
    err = cudaMemcpy(d_B, B.data(), size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         std::cerr << "Error copying B to device: " << cudaGetErrorString(err) << std::endl;
         cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
         return;
    }
    
    // Set up the execution configuration
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;
    
    if (blockSize == 1024) {
        threadsPerBlock = dim3(32, 32);
        blocksPerGrid = dim3((c_cols + 31) / 32, (c_rows + 31) / 32);
    }
    else if (blockSize == 2048) {
	threadsPerBlock = dim3(32, 64);
	blocksPerGrid = dim3((c_cols + 31) / 32, (c_rows + 63) / 64);
    }
    else if (blockSize == 512) {
	threadsPerBlock = dim3(16, 32);
	blocksPerGrid = dim3((c_cols + 15) / 16, (c_rows + 31) / 32);
    }
    else {
        std::cerr << "Invalid block size: " << blockSize << "\n";
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return;
    }
    
    // Launch the kernel.
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
         std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
         cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
         return;
    }
    
    // Wait for the kernel to finish.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
         std::cerr << "cudaDeviceSynchronize error: " << cudaGetErrorString(err) << std::endl;
         cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
         return;
    }
    
    // Copy the result matrix back to host memory.
    err = cudaMemcpy(C.data(), d_C, size_c, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
         std::cerr << "Error copying C from device: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Free device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Sequential matrix multiplication: C = A * B
void sequentialMultiply(const std::vector<double> &A, const std::vector<double> &B,
                        std::vector<double> &C, int M, int numThreads, int blockSize) {
    // numThreads is not used in this function, but is included to allow hot-swapping
    // determine row and column sizes based on matrices, not M
    int rowSize = A.size() / M;
    int colSize = B.size() / M;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// Worker function for multithreaded multiplication.
void threadMultiplyWorker(const std::vector<double>& A, const std::vector<double>& B,
                          std::vector<double>& C, int M, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < M; j++) {
            double sum = 0.0;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// Multithreaded multiplication: divides rows among the specified number of threads.
void multithreadedMultiply(const std::vector<double> &A, const std::vector<double> &B,
                           std::vector<double> &C, int M, int numThreads, int blockSize) {
    std::vector<std::thread> threads;
    int rowsToProcess = A.size() / M;
    int rowsPerThread = rowsToProcess / numThreads;
    int extraRows = rowsToProcess % numThreads;
    int startRow = 0;
    for (int t = 0; t < numThreads; t++) {
        int endRow = startRow + rowsPerThread + (t < extraRows ? 1 : 0);
        threads.emplace_back(threadMultiplyWorker, std::cref(A), std::cref(B),
                             std::ref(C), M, startRow, endRow);
        startRow = endRow;
    }
    for (auto &th : threads) {
        th.join();
    }
}

void openmpMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, int M, int numThreads, int blockSize) {
    // Parallelize the outer loop using OpenMP, with a specified number of threads.
    #pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            double sum = 0.0;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// void distributedMultiply(const std::vector<double>& A, const std::vector<double>& B,
//                          std::vector<double>& C, int M, void (*localMultiplyFunc)(const std::vector<double>&, const std::vector<double>&, std::vector<double>&, int, int, int),
// 			 int numThreads, int cudaBlockSize) {
//
//     int worldRank, worldSize;
//     MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
//     MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
//
//     if (M % worldSize != 0) {
//         if (worldRank == 0)
//             std::cerr << "Error: Matrix size M must be divisible by the number of processes.\n";
//         MPI_Abort(MPI_COMM_WORLD, 1);
//     }
//
//     int blockSize = M / worldSize;
//     std::vector<double> localA(blockSize * M);
//     std::vector<double> localC(blockSize * M);
//
//     // Scatter A (each process gets a block of rows)
//     MPI_Scatter(A.data(), blockSize * M, MPI_DOUBLE,
//                 localA.data(), blockSize * M, MPI_DOUBLE,
//                 0, MPI_COMM_WORLD);
//
//     // Instead of scattering B, broadcast B to all processes
//     std::vector<double> fullB = B;  // Ensure B is of size M*M on the root
//     MPI_Bcast(fullB.data(), M * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//     // Compute local block of C
//     localMultiplyFunc(localA, fullB, localC, M, numThreads, cudaBlockSize);
//
//
//     // Gather local results into C on the root process
//     MPI_Gather(localC.data(), blockSize * M, MPI_DOUBLE,
//                C.data(), blockSize * M, MPI_DOUBLE,
//                0, MPI_COMM_WORLD);
//
//     // Synchronize processes
//     MPI_Barrier(MPI_COMM_WORLD);
// }


// distributedMultiply uses Cannon's algorithm to perform distributed matrix multiplication.
// A and B are M×M matrices stored in row-major order, and C will store the result (only on root).
// localMultiplyFunc performs the multiplication of two square submatrices of size 'blockSize' (and may use threads or CUDA).
void distributedMultiply(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, int M,
                         void (*localMultiplyFunc)(const std::vector<double>&,
                                                   const std::vector<double>&,
                                                   std::vector<double>&, int, int, int),
			 int numThreads, int cudaBlockSize) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Cannon's algorithm requires a square grid of processes.
    int sqrtP = static_cast<int>(std::sqrt(worldSize));
    if (sqrtP * sqrtP != worldSize) {
        if (worldRank == 0)
            std::cerr << "Error: Number of processes (" << worldSize 
                      << ") must be a perfect square for Cannon's algorithm.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Ensure the matrix dimension is divisible by the grid dimension.
    if (M % sqrtP != 0) {
        if (worldRank == 0)
            std::cerr << "Error: Matrix size M (" << M 
                      << ") must be divisible by sqrt(number of processes) (" << sqrtP << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int blockSize = M / sqrtP;

    // Create a 2D Cartesian communicator with periodic boundaries.
    int dims[2] = {sqrtP, sqrtP};
    int periods[2] = {1, 1};
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartComm);

    int coords[2];
    MPI_Cart_coords(cartComm, worldRank, 2, coords);

    // Allocate local submatrices: each is blockSize x blockSize.
    std::vector<double> localA(blockSize * blockSize);
    std::vector<double> localB(blockSize * blockSize);
    std::vector<double> localC(blockSize * blockSize, 0.0);

    // --------------------
    // Scatter submatrices
    // --------------------
    // For Cannon's algorithm we need a 2D block decomposition.
    // Here, the root (rank 0) extracts each block from A and B and sends it to the appropriate process.
    if (worldRank == 0) {
        // Scatter A: send each block to the corresponding process.
        for (int proc = 0; proc < worldSize; proc++) {
            int procCoords[2] = {proc / sqrtP, proc % sqrtP};
            std::vector<double> block(blockSize * blockSize);
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int globalRow = procCoords[0] * blockSize + i;
                    int globalCol = procCoords[1] * blockSize + j;
                    block[i * blockSize + j] = A[globalRow * M + globalCol];
                }
            }
            if (proc == 0)
                localA = block;
            else
                MPI_Send(block.data(), blockSize * blockSize, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }
        // Scatter B similarly.
        for (int proc = 0; proc < worldSize; proc++) {
            int procCoords[2] = {proc / sqrtP, proc % sqrtP};
            std::vector<double> block(blockSize * blockSize);
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int globalRow = procCoords[0] * blockSize + i;
                    int globalCol = procCoords[1] * blockSize + j;
                    block[i * blockSize + j] = B[globalRow * M + globalCol];
                }
            }
            if (proc == 0)
                localB = block;
            else
                MPI_Send(block.data(), blockSize * blockSize, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(localA.data(), blockSize * blockSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localB.data(), blockSize * blockSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ---------------------------------------------------------
    // Initial alignment: skew localA left and localB up.
    // ---------------------------------------------------------
    // Each process (i, j) shifts A left by i positions.
    for (int i = 0; i < coords[0]; i++) {
        int src, dst;
        MPI_Cart_shift(cartComm, 1, -1, &src, &dst);
        MPI_Sendrecv_replace(localA.data(), blockSize * blockSize, MPI_DOUBLE,
                             dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
    }
    // Each process (i, j) shifts B up by j positions.
    for (int i = 0; i < coords[1]; i++) {
        int src, dst;
        MPI_Cart_shift(cartComm, 0, -1, &src, &dst);
        MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_DOUBLE,
                             dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
    }

    // -------------------------
    // Main Cannon's algorithm loop
    // -------------------------
    // There will be sqrtP steps. At each step, multiply the local submatrices
    // and then shift A left and B up by one.
    for (int step = 0; step < sqrtP; step++) {
        // Multiply local submatrices and accumulate into localC.
        // Note: We pass blockSize as the matrix dimension for the local multiplication.
        localMultiplyFunc(localA, localB, localC, blockSize, numThreads, cudaBlockSize);

        // Shift localA one step to the left.
        {
            int src, dst;
            MPI_Cart_shift(cartComm, 1, -1, &src, &dst);
            MPI_Sendrecv_replace(localA.data(), blockSize * blockSize, MPI_DOUBLE,
                                 dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
        }
        // Shift localB one step upward.
        {
            int src, dst;
            MPI_Cart_shift(cartComm, 0, -1, &src, &dst);
            MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_DOUBLE,
                                 dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
        }
    }

    // ---------------------------------------------------------
    // Gather the localC blocks back into the global C matrix.
    // ---------------------------------------------------------
    if (worldRank == 0) {
        C.resize(M * M, 0.0);
        // Place the block from process 0.
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                C[i * M + j] = localC[i * blockSize + j];
            }
        }
        // Receive blocks from the other processes.
        for (int proc = 1; proc < worldSize; proc++) {
            int procCoords[2] = {proc / sqrtP, proc % sqrtP};
            std::vector<double> block(blockSize * blockSize);
            MPI_Recv(block.data(), blockSize * blockSize, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    int globalRow = procCoords[0] * blockSize + i;
                    int globalCol = procCoords[1] * blockSize + j;
                    C[globalRow * M + globalCol] = block[i * blockSize + j];
                }
            }
        }
    } else {
        MPI_Send(localC.data(), blockSize * blockSize, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&cartComm);
}


// Structure to hold timing results
struct Timing {
    double t_reading;
    double t_multiplication;
    double t_writing;
    double t_total;
};

enum VariantType { V1, V2A, V2B, V3A, V3B, V4A, V4B, V5A, V5B, V6, V7A, V7B };


// Run one experiment iteration using the given multiplication function.
Timing runExperiment(const std::string &fileA, const std::string &fileB, const std::string &fileC,
                     int M, VariantType var, int numThreads = 1, int blockSize = 1024) {
    Timing t{0, 0, 0, 0};
    std::vector<double> A, B, C(M * M);
    auto startTotal = std::chrono::high_resolution_clock::now();

    // Reading phase
    auto startRead = std::chrono::high_resolution_clock::now();
    if (var == V1) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V2A) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V2B) {
	    parallelReadMatrix(fileA, A, M, numThreads);
	    parallelReadMatrix(fileB, B, M, numThreads);
    } else if (var == V3A) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V3B) {
	    parallelReadMatrix(fileA, A, M, numThreads);
	    parallelReadMatrix(fileB, B, M, numThreads);
    } else if (var == V4A) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V4B) {
	    parallelReadMatrix(fileA, A, M, numThreads);
	    parallelReadMatrix(fileB, B, M, numThreads);
    } else if (var == V5A) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V5B) {
	    parallelReadMatrix(fileA, A, M, numThreads);
	    parallelReadMatrix(fileB, B, M, numThreads);
    } else if (var == V6) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V7A) {
	    sequentialReadMatrix(fileA, A, M);
	    sequentialReadMatrix(fileB, B, M);
    } else if (var == V7B) {
	    parallelReadMatrix(fileA, A, M, numThreads);
	    parallelReadMatrix(fileB, B, M, numThreads);
    }
	auto endRead = std::chrono::high_resolution_clock::now();
    t.t_reading = std::chrono::duration<double>(endRead - startRead).count();

    // Multiplication phase
    auto startMul = std::chrono::high_resolution_clock::now();
    if (var == V1) {
    	sequentialMultiply(A, B, C, M, numThreads, blockSize);
    } else if (var == V2A) {
        multithreadedMultiply(A, B, C, M, numThreads, blockSize);
    } else if (var == V2B) {
        sequentialMultiply(A, B, C, M, numThreads, blockSize);
    } else if (var == V3A) {
	openmpMultiply(A, B, C, M, numThreads, blockSize);
    } else if (var == V3B) {
	openmpMultiply(A, B, C, M, numThreads, blockSize);
    } else if (var == V4A) {
	distributedMultiply(A, B, C, M, &sequentialMultiply, numThreads, blockSize);
    } else if (var == V4B) {
	distributedMultiply(A, B, C, M, &sequentialMultiply, numThreads, blockSize);
    } else if (var == V5A) {
	distributedMultiply(A, B, C, M, &multithreadedMultiply, numThreads, blockSize);
    } else if (var == V5B) {
	distributedMultiply(A, B, C, M, &multithreadedMultiply, numThreads, blockSize);
    } else if (var == V6) {
	cudaMultiply(A, B, C, M, numThreads, blockSize);
    } else if (var == V7A) {
	distributedMultiply(A, B, C, M, &cudaMultiply, numThreads, blockSize);
    } else if (var == V7B) {
	distributedMultiply(A, B, C, M, &cudaMultiply, numThreads, blockSize);
    }
    auto endMul = std::chrono::high_resolution_clock::now();
    t.t_multiplication = std::chrono::duration<double>(endMul - startMul).count();

    // Writing phase (output in plaintext)
	// only write the result if process is rank 0
    auto startWrite = std::chrono::high_resolution_clock::now();
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
	// std::cout << "Writing output if needed, world rank: " << worldRank << "\n";
	if (worldRank == 0) {
		std::cout << "Entering write phase\n";
		writeMatrix(fileC, C, M);
		std::cout << "Output written to " << fileC << "\n";
	}

	MPI_Barrier(MPI_COMM_WORLD);

    auto endWrite = std::chrono::high_resolution_clock::now();
    t.t_writing = std::chrono::duration<double>(endWrite - startWrite).count();

    auto endTotal = std::chrono::high_resolution_clock::now();
    t.t_total = std::chrono::duration<double>(endTotal - startTotal).count();
    return t;
}

// Helper: write experiment results to JSON file.
void writeJSON(const std::string &jsonFile, const std::string &variantName, double avgReading,
               double avgMul, double avgWrite, double avgTotal) {
    std::ofstream fout(jsonFile);
    if (!fout) {
        throw std::runtime_error("Cannot open file " + jsonFile + " for writing JSON output.");
    }
    fout << "{\n";
    fout << "  \"T_reading\": " << avgReading << ",\n";
    fout << "  \"T_multiplication\": " << avgMul << ",\n";
    fout << "  \"T_writing\": " << avgWrite << ",\n";
    fout << "  \"T_total\": " << avgTotal << "\n";
    fout << "}\n";
    fout.close();
}

// Print usage help message.
void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " --variant <variant> --M <matrix_size> [--threads <numThreads>]\n";
    std::cerr << "  <variant> can be \"1\" for sequential or \"2a\" (or similar) for multithreaded.\n";
    std::cerr << "  For multithreaded variant, --threads must be specified (e.g., 10, 20, or 50).\n";
    std::cerr << "  For CUDA variant, --blockSize must be specified (e.g., 1024 or 4096).\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);

    std::string variantStr, fileA, fileB, fileC, jsonFile;
    int M = 0;
    int numThreads = 1;
    int blockSize = 1;
    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            variantStr = argv[++i];
        } else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            numThreads = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--blockSize") == 0 && i + 1 < argc) {
	    blockSize = std::atoi(argv[++i]);
	}
    }

    fileA = "A_M" + std::to_string(M) + ".bin";
	fileB = "B_M" + std::to_string(M) + ".bin";
	fileC = "C_V" + variantStr + "_M" + std::to_string(M) + "_T" + std::to_string(numThreads) + "_B" + std::to_string(blockSize) + ".txt";
	jsonFile = variantStr + "_M" + std::to_string(M) + "_T" + std::to_string(numThreads) + "_B" + std::to_string(blockSize) + ".json";

    if (M <= 0 || variantStr.empty() || fileA.empty() || fileB.empty() || fileC.empty() || jsonFile.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    // Determine the variant type
    VariantType currentVariant = V1;
    if (variantStr == "1") {
        currentVariant = V1;
    } else if (variantStr == "2a") {
			currentVariant = V2A;
    } else if (variantStr == "2b") {
			currentVariant = V2B;
    } else if (variantStr == "3a") {
			currentVariant = V3A;
    } else if (variantStr == "3b") {
			currentVariant = V3B;
    } else if (variantStr == "4a") {
			currentVariant = V4A;
    } else if (variantStr == "4b") {
			currentVariant = V4B;
    } else if (variantStr == "5a") {
			currentVariant = V5A;
    } else if (variantStr == "5b") {
			currentVariant = V5B;
    } else if (variantStr == "6") {
			currentVariant = V6;
    } else if (variantStr == "7a") {
	    		currentVariant = V7A;
    } else if (variantStr == "7b") {
	    		currentVariant = V7B;
    } else {
	std::cerr << "Unknown variant: " << variantStr << "\n";
	return 1;
    }

    // Generate the input matrices if they do not exist.
    try {
        if (!fileExists(fileA)) {
            std::cout << "Generating matrix A and saving to " << fileA << "\n";
            generateMatrix(fileA, M);
        }
	else {
	    std::cout << "Matrix A already exists, skipping generation.\n";
	}
        if (!fileExists(fileB)) {
            std::cout << "Generating matrix B and saving to " << fileB << "\n";
            generateMatrix(fileB, M);
        }
	else {
	    std::cout << "Matrix B already exists, skipping generation.\n";
	}
    } catch (const std::exception &e) {
        std::cerr << "Error during matrix generation: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Running experiment with variant " << variantStr << " and matrix size " << M << "\n";


    // Run 10 executions for the chosen variant
    const int numRuns = 10;
    double totalReading = 0.0, totalMultiplication = 0.0, totalWriting = 0.0, totalTotal = 0.0;
    for (int run = 0; run < numRuns; run++) {
	if (worldRank == 0) {
		std::cout << "Run " << run + 1 << " of " << numRuns << "\n";
	}
        Timing t = runExperiment(fileA, fileB, fileC, M, currentVariant, numThreads, blockSize);
        totalReading += t.t_reading;
        totalMultiplication += t.t_multiplication;
        totalWriting += t.t_writing;
        totalTotal += t.t_total;
    }
    double avgReading = totalReading / numRuns;
    double avgMultiplication = totalMultiplication / numRuns;
    double avgWriting = totalWriting / numRuns;
    double avgTotal = totalTotal / numRuns;


    // If variant is different than 1, load matrix C and compare it with the result.
    if (worldRank == 0 && currentVariant != V1) {
	std::vector<double> C1(M * M), C2(M * M);
	try {
	    auto benchFileC = "C_V1_M" + std::to_string(M) + "_T" + std::to_string(numThreads) + "_B" + std::to_string(blockSize) + ".txt";
	    sequentialReadMatrix(fileC, C1, M);
	    sequentialReadMatrix(fileC, C2, M);
	} catch (const std::exception &e) {
	    std::cerr << "Error reading matrix C: " << e.what() << "\n";
	    return 1;
	}
	if (C1 != C2) {
	    std::cerr << "Error: Matrices C1 and C2 are different.\n";
	    return 1;
	}

	std::cout << "PASS Matrices C1 and C2 are equal.\n";
    }

    // Output the averaged timings and speed-up in JSON format.
    auto startWrite = std::chrono::high_resolution_clock::now();
	if (worldRank == 0) {
			try {
				writeJSON(jsonFile, variantStr, avgReading, avgMultiplication, avgWriting, avgTotal);
				std::cout << "Experiment completed. Results written to " << jsonFile << "\n";
			} catch (const std::exception &e) {
				std::cerr << "Error writing JSON output: " << e.what() << "\n";
				return 1;
			}
    }

    MPI_Finalize();
    return 0;
}

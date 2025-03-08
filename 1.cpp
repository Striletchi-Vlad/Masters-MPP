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

// Sequential matrix multiplication: C = A * B
void sequentialMultiply(const std::vector<double> &A, const std::vector<double> &B,
                        std::vector<double> &C, int M) {
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
                           std::vector<double> &C, int M, int numThreads) {
    std::vector<std::thread> threads;
    int rowsPerThread = M / numThreads;
    int extraRows = M % numThreads;
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
                    std::vector<double>& C, int M, int numThreads) {
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

// MPI implementation of matrix multiplication using Cannon's algorithm.
// A, B, and C are stored as row-major 1D arrays of size M*M.
// It is assumed that M is divisible by Q = sqrt(usedProcs),
// where usedProcs = Q*Q and Q = floor(sqrt(total MPI processes)).
void mpiCannonMultiply(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>& C, int M) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Determine grid dimensions from the total processes.
    int gridDim = static_cast<int>(std::floor(std::sqrt(worldSize)));
    int usedProcs = gridDim * gridDim; // Only these many processes participate.

    // Create a new group and communicator for the used processes.
    MPI_Group worldGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    std::vector<int> gridRanks(usedProcs);
    for (int i = 0; i < usedProcs; i++) {
        gridRanks[i] = i;  // use the first usedProcs ranks
    }
    MPI_Group gridGroup;
    MPI_Group_incl(worldGroup, usedProcs, gridRanks.data(), &gridGroup);
    MPI_Comm gridComm;
    MPI_Comm_create(MPI_COMM_WORLD, gridGroup, &gridComm);

    // Processes not in the grid communicator simply exit.
    if (gridComm == MPI_COMM_NULL) {
        MPI_Group_free(&worldGroup);
        MPI_Group_free(&gridGroup);
        return;
    }

    // Create a 2D Cartesian communicator for the grid.
    int dims[2]    = { gridDim, gridDim };
    int periods[2] = { 1, 1 }; // wrap-around in both dimensions for Cannon's shifts
    int reorder    = 1;
    MPI_Comm cartComm;
    MPI_Cart_create(gridComm, 2, dims, periods, reorder, &cartComm);

    int cartRank;
    MPI_Comm_rank(cartComm, &cartRank);
    int coords[2];
    MPI_Cart_coords(cartComm, cartRank, 2, coords);

    // Compute the local block size.
    // (Assumes M is divisible by gridDim.)
    int blockSize = M / gridDim;

    // Allocate local blocks for A, B, and the result C.
    std::vector<double> localA(blockSize * blockSize);
    std::vector<double> localB(blockSize * blockSize);
    std::vector<double> localC(blockSize * blockSize, 0.0);

    // Create a derived datatype that describes a block (submatrix) of size blockSize x blockSize
    // in an M x M matrix stored in row-major order.
    MPI_Datatype blockType, blockTypeResized;
    MPI_Type_vector(blockSize, blockSize, M, MPI_DOUBLE, &blockType);
    MPI_Type_create_resized(blockType, 0, blockSize * sizeof(double), &blockTypeResized);
    MPI_Type_commit(&blockTypeResized);
    MPI_Type_free(&blockType); // no longer needed

    // Prepare parameters for scattering the full matrices.
    // Only the root (cartComm rank 0) holds the full matrices.
    std::vector<int> sendCounts, displs;
    if (cartRank == 0) {
        sendCounts.resize(usedProcs, 1);
        displs.resize(usedProcs);
        // For each block, compute the displacement in the global matrix.
        // The starting element of the block at grid position (i, j) is at row i*blockSize and col j*blockSize.
        for (int i = 0; i < gridDim; i++) {
            for (int j = 0; j < gridDim; j++) {
                displs[i * gridDim + j] = i * M + j * blockSize;
            }
        }
    }

    // Scatter the blocks of matrix A into localA.
    MPI_Scatterv((cartRank == 0 ? A.data() : nullptr),
                 (cartRank == 0 ? sendCounts.data() : nullptr),
                 (cartRank == 0 ? displs.data() : nullptr),
                 blockTypeResized,
                 localA.data(), blockSize * blockSize, MPI_DOUBLE,
                 0, cartComm);

    // Scatter the blocks of matrix B into localB.
    MPI_Scatterv((cartRank == 0 ? B.data() : nullptr),
                 (cartRank == 0 ? sendCounts.data() : nullptr),
                 (cartRank == 0 ? displs.data() : nullptr),
                 blockTypeResized,
                 localB.data(), blockSize * blockSize, MPI_DOUBLE,
                 0, cartComm);

    // Initial alignment for Cannon's algorithm:
    // Shift localA left by its row coordinate and localB up by its column coordinate.
    int src, dst;
    // For A: shift left by coords[0]
    MPI_Cart_shift(cartComm, 1, -coords[0], &src, &dst);
    MPI_Sendrecv_replace(localA.data(), blockSize * blockSize, MPI_DOUBLE,
                         dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
    // For B: shift up by coords[1]
    MPI_Cart_shift(cartComm, 0, -coords[1], &src, &dst);
    MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_DOUBLE,
                         dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);

    // Main loop of Cannon's algorithm.
    for (int step = 0; step < gridDim; step++) {
        // Multiply local blocks and accumulate into localC.
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                for (int k = 0; k < blockSize; k++) {
                    localC[i * blockSize + j] += localA[i * blockSize + k] * localB[k * blockSize + j];
                }
            }
        }
        // Shift localA left by one.
        MPI_Cart_shift(cartComm, 1, -1, &src, &dst);
        MPI_Sendrecv_replace(localA.data(), blockSize * blockSize, MPI_DOUBLE,
                             dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
        // Shift localB up by one.
        MPI_Cart_shift(cartComm, 0, -1, &src, &dst);
        MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_DOUBLE,
                             dst, 0, src, 0, cartComm, MPI_STATUS_IGNORE);
    }

    // Gather the computed localC blocks to the root (cartComm rank 0).
    std::vector<double> gatheredC;
    if (cartRank == 0) {
        gatheredC.resize(M * M, 0.0);
    }
    MPI_Gatherv(localC.data(), blockSize * blockSize, MPI_DOUBLE,
                (cartRank == 0 ? gatheredC.data() : nullptr),
                (cartRank == 0 ? sendCounts.data() : nullptr),
                (cartRank == 0 ? displs.data() : nullptr),
                blockTypeResized,
                0, cartComm);

    // Now, have the rank 0 process of MPI_COMM_WORLD (which is also cartComm rank 0)
    // copy the gathered result into C.
    if (worldRank == 0 && cartRank == 0) {
        C = gatheredC;
    }

    // Clean up.
    MPI_Type_free(&blockTypeResized);
    MPI_Comm_free(&cartComm);
    MPI_Comm_free(&gridComm);
    MPI_Group_free(&gridGroup);
    MPI_Group_free(&worldGroup);
}

// Structure to hold timing results
struct Timing {
    double t_reading;
    double t_multiplication;
    double t_writing;
    double t_total;
};

enum VariantType { V1, V2A, V2B, V3A, V3B, V4A };


// Run one experiment iteration using the given multiplication function.
Timing runExperiment(const std::string &fileA, const std::string &fileB, const std::string &fileC,
                     int M, VariantType var, int numThreads = 1) {
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
    }
	auto endRead = std::chrono::high_resolution_clock::now();
    t.t_reading = std::chrono::duration<double>(endRead - startRead).count();

    // Multiplication phase
    auto startMul = std::chrono::high_resolution_clock::now();
    if (var == V1) {
        sequentialMultiply(A, B, C, M);
    } else if (var == V2A) {
        multithreadedMultiply(A, B, C, M, numThreads);
    } else if (var == V2B) {
        sequentialMultiply(A, B, C, M);
    } else if (var == V3A) {
			openmpMultiply(A, B, C, M, numThreads);
    } else if (var == V3B) {
			openmpMultiply(A, B, C, M, numThreads);
    } else if (var == V4A) {
			mpiCannonMultiply(A, B, C, M);
    }
    auto endMul = std::chrono::high_resolution_clock::now();
    t.t_multiplication = std::chrono::duration<double>(endMul - startMul).count();

    // Writing phase (output in plaintext)
	// only write the result if process is rank 0
    auto startWrite = std::chrono::high_resolution_clock::now();
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
	std::cout << "Writing output if needed, world rank: " << worldRank << "\n";
	if (worldRank == 0) {
			writeMatrix(fileC, C, M);
			std::cout << "Output written to " << fileC << "\n";
    }
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
    fout << "  \"Var\": \"" << variantName << "\",\n";
    fout << "  \"T_reading\": " << avgReading << ",\n";
    fout << "  \"T_multiplication\": " << avgMul << ",\n";
    fout << "  \"T_writing\": " << avgWrite << ",\n";
    fout << "  \"T_total\": " << avgTotal << ",\n";
    fout << "}\n";
    fout.close();
}

// Print usage help message.
void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " --variant <variant> --M <matrix_size> [--threads <numThreads>]\n";
    std::cerr << "  <variant> can be \"1\" for sequential or \"2a\" (or similar) for multithreaded.\n";
    std::cerr << "  For multithreaded variant, --threads must be specified (e.g., 10, 20, or 50).\n";
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
    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            variantStr = argv[++i];
        } else if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            numThreads = std::atoi(argv[++i]);
        }
    }

    fileA = "A_M" + std::to_string(M) + ".bin";
	fileB = "B_M" + std::to_string(M) + ".bin";
	fileC = "C_V" + variantStr + "_M" + std::to_string(M) + ".txt"; 
	jsonFile = variantStr + "_M" + std::to_string(M) + "_T" + std::to_string(numThreads) + ".json";

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
    // const int numRuns = 10;
    const int numRuns = 1;
    double totalReading = 0.0, totalMultiplication = 0.0, totalWriting = 0.0, totalTotal = 0.0;
    for (int run = 0; run < numRuns; run++) {
	std::cout << "Run " << run + 1 << " of " << numRuns << "\n";
        Timing t = runExperiment(fileA, fileB, fileC, M, currentVariant, numThreads);
        totalReading += t.t_reading;
        totalMultiplication += t.t_multiplication;
        totalWriting += t.t_writing;
        totalTotal += t.t_total;
    }
    double avgReading = totalReading / numRuns;
    double avgMultiplication = totalMultiplication / numRuns;
    double avgWriting = totalWriting / numRuns;
    double avgTotal = totalTotal / numRuns;

    // Output the averaged timings and speed-up in JSON format.
    auto startWrite = std::chrono::high_resolution_clock::now();
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
	std::cout << "Writing JSON output if needed, world rank: " << worldRank << "\n";
	if (worldRank == 0) {
			try {
				writeJSON(jsonFile, variantStr, avgReading, avgMultiplication, avgWriting, avgTotal);
				std::cout << "Experiment completed. Results written to " << jsonFile << "\n";
			} catch (const std::exception &e) {
				std::cerr << "Error writing JSON output: " << e.what() << "\n";
				return 1;
			}
    }

    return 0;
}

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

// Structure to hold timing results
struct Timing {
    double t_reading;
    double t_multiplication;
    double t_writing;
    double t_total;
};

enum VariantType { V1, V2A, V2B };


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
    }
    auto endMul = std::chrono::high_resolution_clock::now();
    t.t_multiplication = std::chrono::duration<double>(endMul - startMul).count();

    // Writing phase (output in plaintext)
    auto startWrite = std::chrono::high_resolution_clock::now();
    writeMatrix(fileC, C, M);
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
	fileC = "C_M" + std::to_string(M) + ".txt";
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
    try {
        writeJSON(jsonFile, variantStr, avgReading, avgMultiplication, avgWriting, avgTotal);
        std::cout << "Experiment completed. Results written to " << jsonFile << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error writing JSON output: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

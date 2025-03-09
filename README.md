## Theoretical Performance Analysis for Matrix Multiplication Variants
**1. Sequential (Variant 1)**:
- **Time Complexity**: The sequential multiplication of two MxM matrices has a time complexity of **O(M^3)**. Reading the two input matrices and writing the output matrix sequentially takes **O(M^2)** each. Therefore, the total time complexity is **O(M^3 + 2M^2)**, which is dominated by the multiplication, resulting in **O(M^3)**.
- **Speed-up**: Speed-up (S) is defined as the ratio of the sequential execution time (T_s) to the parallel execution time (T_p). For the sequential variant itself, the speed-up is trivially **1** (S = T_s / T_s = 1).
- **Efficiency**: Efficiency (E) is the speed-up per processing element. With one processing element, the efficiency is **1** (E = S / 1 = 1).
- **Cost**: Cost (C) is the product of the parallel execution time and the number of processing elements. For the sequential variant, the cost is **O(M^3) * 1 = O(M^3)**.

**2. Parallel - explicit multithreading (Variants 2a and 2b)**:
- Number of Threads (p): 10, 20, 50.
- **2a: Sequential reading, parallel multiplication, sequential writing.**
    - **Time Complexity**: Sequential reading takes **O(M^2)**. Parallel multiplication using p threads can ideally achieve a time complexity of **O(M^3 / p)**. Sequential writing takes **O(M^2)**. The overall time complexity is **O(M^2 + M^3 / p)**. For large M and sufficient p, the multiplication term dominates.
    - **Speed-up**: S = T_s / T_p = O(M^3) / O(M^2 + M^3 / p). If M is large and p is significant, Speed-up ≈ **p**.
    - **Efficiency**: E = S / p ≈ (O(M^3) / (M^2 + M^3 / p)) / p = O(M^3) / (pM^2 + M^3). If M is large, Efficiency ≈ **1**.
    - **Cost**: C = T_p * p = O(M^2 + M^3 / p) * p = **O(pM^2 + M^3)**.
- **2b: Parallel reading, sequential multiplication, sequential writing.**
    - **Time Complexity**: Parallel reading with p threads can potentially reduce the reading time to **O(M^2 / p)**. Sequential multiplication remains **O(M^3)**. Sequential writing takes **O(M^2)**. The overall time complexity is **O(M^2 / p + M^3 + M^2) ≈ O(M^3)**, as the sequential multiplication dominates.
    - **Speed-up**: S = T_s / T_p = O(M^3) / O(M^3 + M^2 / p). The speed-up will be slightly larger than 1 if the reading time was a significant portion of the sequential time. Generally, Speed-up will be **>= 1**.
    - **Efficiency**: E = S / p ≈ (O(M^3) / (M^3 + M^2 / p)) / p = O(M^3) / (pM^3 + M^2). Efficiency will be **O(1/p)**.
    - **Cost**: C = T_p * p = O(M^3 + M^2 / p) * p = **O(pM^3 + M^2)**.

**3. Parallel - OpenMP (Variants 3a and 3b)**:
- Number of Threads (p): 10, 20, 50.
- **3a: Sequential reading, parallel multiplication, sequential writing.**
    - The theoretical performance analysis is the same as Variant 2a because it involves sequential reading and writing with parallel multiplication using p threads.
    - **Time Complexity**: **O(M^2 + M^3 / p)**.
    - **Speed-up**: S ≈ **p** (for large M).
    - **Efficiency**: E ≈ **1** (for large M).
    - **Cost**: **O(pM^2 + M^3)**.
- **3b: Parallel reading, parallel multiplication, sequential writing.**
    - **Time Complexity**: Parallel reading with p threads: **O(M^2 / p)**. Parallel multiplication with p threads: **O(M^3 / p)**. Sequential writing: **O(M^2)**. The overall time complexity is **O(M^2 / p + M^3 / p + M^2)**.
    - **Speed-up**: S = O(M^3) / O(M^2 / p + M^3 / p + M^2). For large M, Speed-up ≈ **p**.
    - **Efficiency**: E = S / p ≈ (O(M^3) / (M^2 / p + M^3 / p + M^2)) / p = O(M^3) / (M^2 + M^3 + pM^2). For large M, Efficiency ≈ **1**.
    - **Cost**: C = T_p * p = O(M^2 / p + M^3 / p + M^2) * p = **O(M^2 + M^3 + pM^2)**.

**4. Parallel - MPI (4 cluster nodes) – using row distribution (Variants 4a and 4b)**:
- Number of Processes (p): 20, 40. These processes are distributed across 4 cluster nodes.

- **4a: Sequential reading, parallel multiplication (row chunks), sequential writing.**
    - **Time Complexity**:
        - **Sequential Reading**: Reading FileA and FileB sequentially takes **O(M^2)** for each, totaling **O(M^2)** as the dominant term.
        - **Parallel Multiplication**: The first matrix A is conceptually divided into p (number of processes) row chunks, with each process receiving approximately M/p rows. The entire matrix B is needed by each process to perform the local multiplication.
            - **Communication (Matrix Distribution)**: The root process (typically rank 0) reads both matrices. It then needs to distribute the row chunks of A to the other p-1 processes and broadcast matrix B to all processes. Scattering approximately M^2/p elements to each process takes roughly **O(M^2 / p)**. Broadcasting M^2 elements takes roughly **O(M^2)**.
            - **Local Multiplication**: Each of the p processes performs a multiplication of its (M/p) x M sub-matrix of A with the M x M matrix B. This local computation takes **O((M/p) * M * M) = O(M^3 / p)**.
        - **Sequential Writing**: The root process gathers the resulting row chunks of matrix C from all processes, which takes roughly **O(M^2 / p)** per process and **O(M^2)** overall. Finally, the root process writes the entire M x M matrix C to a file sequentially, which takes **O(M^2)**.
        - **Overall Time Complexity**: The dominant terms are the initial sequential reading and the parallel multiplication. Thus, the overall time complexity is approximately **O(M^2 + M^3 / p)**.
    - **Speed-up**: Speed-up (S) = T_sequential / T_parallel = O(M^3) / O(M^2 + M^3 / p).
        - For large M and a significant number of processes p, the M^3/p term dominates the parallel time, and the speed-up approaches **O(M^3) / O(M^3 / p) = O(p)**.
    - **Efficiency**: Efficiency (E) = Speed-up / Number of Processes = (O(M^3) / O(M^2 + M^3 / p)) / p = O(M^3) / (pM^2 + M^3).
        - For large M, the efficiency approaches **O(M^3) / O(M^3) = O(1)**.
    - **Cost**: Cost (C) = T_parallel * Number of Processes = O(M^2 + M^3 / p) * p = **O(pM^2 + M^3)**.

- **4b: Parallel reading, parallel multiplication (row chunks), sequential writing.**
    - **Time Complexity**:
        - **Parallel Reading**: Both FileA and FileB are read in parallel by the p processes. Ideally, if the file reading is perfectly parallelized, this could take **O(M^2 / p)**. However, disk I/O can often be a bottleneck, and the actual speed-up might be less than linear.
        - **Parallel Multiplication**: Similar to Variant 4a, each process multiplies its (M/p) x M chunk of A with the entire M x M matrix B. This takes **O(M^3 / p)**. Since each process might have read its chunk of A directly (if parallel reading is efficient), the explicit scatter communication for A might be reduced. However, matrix B still needs to be available to all, likely through broadcasting (or each process reading it).
        - **Sequential Writing**: The root process gathers the M x M result matrix C (in chunks) which takes approximately **O(M^2)**, and then writes it sequentially in **O(M^2)**.
        - **Overall Time Complexity**: The overall time complexity is approximately **O(M^2 / p + M^3 / p + M^2)**. The **O(M^3 / p)** term for parallel multiplication will likely dominate for large M.
    - **Speed-up**: Speed-up (S) = T_sequential / T_parallel = O(M^3) / O(M^2 / p + M^3 / p + M^2).
        - For large M and significant p, the speed-up approaches **O(M^3) / O(M^3 / p) = O(p)**. The parallel reading might offer a slight improvement over variant 4a if it scales well.
    - **Efficiency**: Efficiency (E) = Speed-up / Number of Processes = (O(M^3) / O(M^2 / p + M^3 / p + M^2)) / p = O(M^3) / (M^2 + M^3 + pM^2).
        - For large M, the efficiency approaches **O(M^3) / O(M^3) = O(1)**.
    - **Cost**: Cost (C) = T_parallel * Number of Processes = O(M^2 / p + M^3 / p + M^2) * p = **O(M^2 + M^3 + pM^2)**.

**5. Parallel - MPI+Multithreading (4 cluster nodes) (Variants 5a and 5b)**:
- Number of Processes (p1): 20, 40.
- Number of Threads per process (p2): 10, 20.
- Total processing elements (p = p1 * p2): 200, 400, 400, 800.
- **5a: Sequential reading, parallel multiplication, sequential writing.**
    - **Time Complexity**: Sequential reading: **O(M^2)**. Parallel multiplication using a hybrid approach (MPI across nodes, multithreading within nodes) with a total of p = p1 \* p2 processing elements can ideally achieve a time complexity of **O(M^3 / (p1 * p2) + communication)**. Sequential writing: **O(M^2)**. Overall: **O(M^2 + M^3 / (p1 * p2) + communication)**.
    - **Speed-up**: S = O(M^3) / O(M^2 + M^3 / (p1 * p2) + communication). For large M and significant p, Speed-up ≈ **p1 * p2**.
    - **Efficiency**: E = S / (p1 * p2) ≈ (O(M^3) / (M^2 + M^3 / (p1 * p2) + communication)) / (p1 * p2) = O(M^3) / ((p1 * p2)M^2 + M^3 + (p1 * p2)communication). For large M, Efficiency ≈ **1**.
    - **Cost**: C = T_p * (p1 * p2) = O(M^2 + M^3 / (p1 * p2) + communication) * (p1 * p2) = **O((p1 * p2)M^2 + M^3 + (p1 * p2)communication)**.

- **5b: Parallel reading, parallel multiplication, sequential writing.**
    -  **Time Complexity**: The initial parallel reading could potentially reduce the reading time to around **O(M^2 / p)** (assuming ideal parallel file system performance), although this might be optimistic. The parallel multiplication remains conceptually around **O(M^3 / (p * t))** for the computational part, plus communication overheads. The sequential writing is **O(M^2)**. Thus, the overall time complexity might be closer to **O(M^2 / p + M^2 (communication) + M^3 / (p * t))**.
    -   **Speed-up**: The parallel reading might lead to a better overall speed-up compared to 5a, especially if file I/O was a significant bottleneck.
    -   **Efficiency**: Similar to 5a, efficiency will depend on load balancing and the overhead of both parallelization methods.
    -   **Cost**: Cost will still be the product of the parallel time and the total number of processing units (*p * t*).

- **6. Parallel - CUDA**
    -   **Time Complexity**: The initial sequential reading is **O(M^2)**. The time complexity of the CUDA multiplication depends on the GPU architecture and the efficiency of the kernel implementation. Ideally, with a large number of parallel threads, it can be significantly faster than CPU-based methods, potentially approaching **O(M^3 / number\_of\_CUDA\_cores)**, assuming sufficient parallelism and memory bandwidth. There is also the overhead of transferring the matrices to the GPU memory and the result back to the host memory. The final sequential writing is **O(M^2)**. Thus, the overall time complexity would be roughly **O(M^2 + data\_transfer\_time + GPU\_multiplication\_time)**.
    -    **Speed-up**: The speed-up can be substantial compared to the sequential CPU version if the matrix size is large enough to effectively utilize the GPU's parallel processing capabilities.
    -   **Efficiency**: Efficiency in this context is often considered relative to the peak performance of the GPU. It depends on how well the kernel utilizes the available resources.
    -   **Cost**: Cost here could be thought of in terms of the time taken multiplied by the "cost" of using the GPU resources.

- **7. Parallel - MPI+CUDA (4 cluster nodes)**
-   **7a: Sequential reading, parallel multiplication, sequential writing**
    -   **Time Complexity**: The initial sequential reading is **O(M^2)**. The parallel multiplication involves MPI communication to distribute work and data, and then GPU-accelerated local computation. If the matrix is divided into 4 roughly equal parts by the MPI processes, each GPU would ideally handle an **O(M^3 / 4)** operation (ignoring communication and data transfer). The overall time would be influenced by the slowest process, which would include communication, data transfer to/from the GPU, and the GPU computation time. The final sequential writing is **O(M^2)**.
    -   **Speed-up**: This approach has the potential for significant speed-up by leveraging both distributed computing and GPU acceleration.
    -   **Efficiency**: Efficiency would consider the utilization of both the CPU cores involved in MPI communication and the GPU resources.
    -   **Cost**: Cost would be the parallel execution time multiplied by the total "cost" of the resources used (CPU time across nodes + GPU time).

-   **7b: Parallel reading, parallel multiplication, sequential writing**
    -   **Time Complexity**: The parallel reading could reduce the initial overhead to roughly **O(M^2 / 4)** (again, assuming good parallel file system performance). The rest of the analysis is similar to 7a, with MPI communication and GPU computation dominating the parallel phase. The overall time complexity might be closer to **O(M^2 / 4 + MPI\_communication\_time + GPU\_computation\_time)**.
    -   **Speed-up**: The parallel reading might offer a further improvement in speed-up over 7a.
    -   **Efficiency**: Efficiency considerations are similar to 7a.
    -   **Cost**: Cost is also similar to 7a.

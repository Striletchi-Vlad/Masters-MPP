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
- **4a: Sequential reading, parallel multiplication (Cannon's), sequential writing.**
  - **Time Complexity:**
    - **Sequential Reading:**  
      Reading FileA and FileB sequentially takes O(M²) for each, totaling O(M²) as the dominant term.
    - **Parallel Multiplication (Cannon's Algorithm):**
      - **Scatter of Submatrices:**  
        The root process scatters _p_ blocks of size (M/sqrt(p)) × (M/sqrt(p)) for both A and B. This takes approximately O(M²/sqrt(p)) to send to each of the (sqrt(p) × sqrt(p) – 1) processes.
      - **Initial Alignment:**  
        Shifting the blocks of A to the left and blocks of B upwards. Each shift involves sending and receiving a block of size (M/sqrt(p))². In the worst case, a block might be shifted (sqrt(p) – 1) times. This contributes a communication cost of roughly O((M/sqrt(p))² × sqrt(p)) = O(M²/sqrt(p)).
      - **Main Loop (sqrt(p) steps):**
        - **Local Multiplication:**  
          Each of the _p_ processes performs a sequential multiplication of two (M/sqrt(p)) × (M/sqrt(p)) submatrices. This takes O((M/sqrt(p))³) = O(M³/p^(3/2)).
        - **Cyclic Shifts:**  
          After each local multiplication, each process shifts its A block to the left and its B block upwards by one position in the grid. This involves sending and receiving a block of size (M/sqrt(p))², taking O((M/sqrt(p))²) = O(M²/p) per step. Over sqrt(p) steps, this becomes O(M²/sqrt(p)).
      - **Gather of Result Submatrices:**  
        The root process gathers the _p_ result blocks of size (M/sqrt(p)) × (M/sqrt(p)) to form the final C matrix. This takes approximately O(M²/sqrt(p)) to receive from each of the (p – 1) processes.
    - **Sequential Writing:**  
      The root process writes the entire M × M matrix C to a file sequentially, which takes O(M²).
    - **Overall Time Complexity:**  
      The dominant terms are the initial sequential reading and the total cost of Cannon's algorithm. The parallel multiplication has a computational cost of  
      _p_ × O(M³/p^(3/2)) × sqrt(p) = O(M³/sqrt(p)).  
      The communication cost is roughly O(M²/sqrt(p)) for scatter, O(M²/sqrt(p)) for initial alignment, and O(M²/sqrt(p)) for cyclic shifts and gather over sqrt(p) steps.  
      Therefore, the overall parallel time complexity is approximately O(M² + M³/sqrt(p) + M²/sqrt(p)). For large M and significant _p_, the O(M³/sqrt(p)) term due to local multiplications dominates.
      
  - **Speed-up:**  
    Speed-up (S) = T<sub>sequential</sub> / T<sub>parallel</sub> = O(M³) / O(M² + M³/sqrt(p) + M²/sqrt(p)).  
    For large M and a significant number of processes _p_, the speed-up approaches O(M³) / O(M³/sqrt(p)) = O(sqrt(p)).
    
  - **Efficiency:**  
    Efficiency (E) = Speed-up / Number of Processes = (O(M³) / O(M² + M³/sqrt(p) + M²/sqrt(p))) / _p_.  
    For large M, the efficiency approaches O(sqrt(p)/p) = O(1/sqrt(p)).
    
  - **Cost:**  
    Cost (C) = T<sub>parallel</sub> × Number of Processes = O(M² + M³/sqrt(p) + M²/sqrt(p)) × _p_ = O(pM² + M³·sqrt(p) + M²·sqrt(p)).

- **4b: Parallel reading, parallel multiplication (Cannon's), sequential writing.**

  - **Time Complexity:**
    - **Parallel Reading:**  
      Both FileA and FileB are read in parallel by the _p_ processes. Ideally, this could take O(M²/p).
    - **Parallel Multiplication (Cannon's Algorithm):**  
      The communication and computation steps remain the same as in Variant 4a: local multiplications in O(M³/p^(3/2)) per step, and sqrt(p) steps, leading to a dominant computation of O(M³/sqrt(p)) across all processes. Communication cost remains roughly O(M²/sqrt(p)).
    - **Sequential Writing:**  
      The root process gathers and writes the result in O(M²).
    - **Overall Time Complexity:**  
      The overall time complexity is approximately O(M²/p + M³/sqrt(p) + M²/sqrt(p)). For large M, the O(M³/sqrt(p)) term for parallel multiplication dominates.
      
  - **Speed-up:**  
    Speed-up (S) = T<sub>sequential</sub> / T<sub>parallel</sub> = O(M³) / O(M²/p + M³/sqrt(p) + M²/sqrt(p)).  
    For large M, the speed-up approaches O(M³) / O(M³/sqrt(p)) = O(sqrt(p)).  
    The parallel reading might offer a slight improvement over Variant 4a if file I/O was a significant bottleneck.
    
  - **Efficiency:**  
    Efficiency (E) = Speed-up / Number of Processes = (O(M³) / O(M²/p + M³/sqrt(p) + M²/sqrt(p))) / _p_.  
    For large M, the efficiency approaches O(1/sqrt(p)).
    
  - **Cost:**  
    Cost (C) = T<sub>parallel</sub> × Number of Processes = O(M²/p + M³/sqrt(p) + M²/sqrt(p)) × _p_ = O(M² + M³·sqrt(p) + M²·sqrt(p)).

**5. Parallel - MPI+Multithreading (4 cluster nodes) (Variants 5a and 5b)**:
- **5a: Sequential reading, parallel multiplication (MPI Cannon's + Multithreading)**
  - **Time Complexity:**
    - **Sequential Reading:**  
      O(M²)
    - **Parallel Multiplication:**  
      Cannon's algorithm is used across p₁ processes, with each process now using p₂ threads to multiply its local (M/√p₁) × (M/√p₁) submatrices.
      - **Communication (MPI):**  
        The communication costs for scatter, initial alignment, and cyclic shifts remain similar to the MPI-only version but involve blocks of size (M/√p₁)². The dominant communication cost is roughly O(M²/√p₁) per step, over √p₁ steps.
      - **Local Multiplication (Multithreaded):**  
        Each of the p₁ processes performs a multiplication of two (M/√p₁) × (M/√p₁) submatrices using p₂ threads. This local computation takes approximately O((M/√p₁)³/p₂) = O(M³/(p₁^(3/2) · p₂)). This happens in √p₁ steps of Cannon's algorithm, so the total parallel computation time is O(M³/(√p₁ · p₂)).
    - **Sequential Writing:**  
      O(M²)
    - **Overall Time Complexity:**  
      Approximately O(M² + M³/(√p₁ · p₂) + M²/√p₁). For large M and significant p, the M³/(√p₁ · p₂) term dominates.
  - **Speed-up:**  
    Speed-up S = (O(M³)) / (O(M² + M³/(√p₁ · p₂) + M²/√p₁)). For large M, S approaches O(M³) / O(M³/(√p₁ · p₂)) = O(√p₁ · p₂).
  - **Efficiency:**  
    Efficiency E = S / (p₁ · p₂) ≈ O(√p₁ · p₂) / (p₁ · p₂) = O(1/√p₁).
  - **Cost:**  
    Cost C = Tₚₐᵣₐₗₗₑₗ × (p₁ · p₂) = O(M² + M³/(√p₁ · p₂) + M²/√p₁) · (p₁ · p₂) = O(p₁ p₂ M² + M³√p₁ + M²√p₁ p₂).

- **5b: Parallel reading, parallel multiplication (MPI Cannon's + Multithreading), sequential writing**
  - **Time Complexity:**
    - **Parallel Reading:**  
      O(M²/p), where p = p₁ × p₂.
    - **Parallel Multiplication:**  
      The computation and communication costs remain the same as in 5a.  
      - Dominant computation: O(M³/(√p₁ · p₂)).  
      - Communication: O(M²/√p₁).
    - **Sequential Writing:**  
      O(M²)
    - **Overall Time Complexity:**  
      Approximately O(M²/(p₁p₂) + M³/(√p₁ · p₂) + M²/√p₁).
  - **Speed-up:**  
    For large M, the speed-up approaches O(√p₁ · p₂), potentially slightly better than 5a due to parallel reading.
  - **Efficiency:**  
    Efficiency remains approximately O(1/√p₁).
  - **Cost:**  
    The cost remains approximately O(p₁ p₂ M² + M³√p₁ + M²√p₁ p₂).

- **6. Parallel - CUDA**
    -   **Time Complexity**: The initial sequential reading is **O(M^2)**. The time complexity of the CUDA multiplication depends on the GPU architecture and the efficiency of the kernel implementation. Ideally, with a large number of parallel threads, it can be significantly faster than CPU-based methods, potentially approaching **O(M^3 / number\_of\_CUDA\_cores)**, assuming sufficient parallelism and memory bandwidth. There is also the overhead of transferring the matrices to the GPU memory and the result back to the host memory. The final sequential writing is **O(M^2)**. Thus, the overall time complexity would be roughly **O(M^2 + data\_transfer\_time + GPU\_multiplication\_time)**.
    -    **Speed-up**: The speed-up can be substantial compared to the sequential CPU version if the matrix size is large enough to effectively utilize the GPU's parallel processing capabilities.
    -   **Efficiency**: Efficiency in this context is often considered relative to the peak performance of the GPU. It depends on how well the kernel utilizes the available resources.
    -   **Cost**: Cost here could be thought of in terms of the time taken multiplied by the "cost" of using the GPU resources.

- **7. Parallel - MPI+CUDA (4 cluster nodes)**

- **7a: Sequential reading, parallel multiplication (MPI Cannon's + CUDA), sequential writing**

  - **Time Complexity:**
    - **Sequential Reading:**  
      O(M²)
    - **Parallel Multiplication:**  
      - **Cannon's Algorithm:**  
        Executed across `p₁` processes.
      - **Communication (MPI):**  
        The matrix is divided into √p₁ × √p₁ blocks.  
        Communication costs for scatter, initial alignment, and cyclic shifts of (M/√p₁) × (M/√p₁) blocks.  
        Each of the √p₁ steps involves communication of O(M²/p₁), so total communication cost is approximately  
        **O(√p₁ × M²/p₁) = O(M²/√p₁)**
      - **Local Multiplication (CUDA):**  
        Each of the `p₁` processes multiplies its local (M/√p₁) × (M/√p₁) submatrices using CUDA.  
        Assuming CUDA provides speedup, the time complexity is  
        **O((M/√p₁)³ / cuda_speedup) + data_transfer_overhead**
        This multiplication occurs over √p₁ steps in Cannon's algorithm.
    - **Sequential Writing:**  
      O(M²)
    - **Overall Time Complexity:**  
      Approximately  
      **O(M² + M²/√p₁ + CUDA_multiplication_time)**  
      For large M, CUDA multiplication time typically dominates.
      
  - **Speed-up:**  
    Significant speed-up over sequential execution due to MPI parallelism and GPU acceleration.  
    Performance scales with both `p₁` and CUDA efficiency.

  - **Efficiency:**  
    Depends on how well the computation and communication are balanced across `p₁` processes, and on CUDA utilization.

  - **Cost:**  
    Parallel execution time × (CPU + GPU resource usage), influenced by `p₁`.

---

- **7b: Parallel reading, parallel multiplication (MPI Cannon's + CUDA), sequential writing**

  - **Time Complexity:**
    - **Parallel Reading:**  
      O(M²/p₁)
    - **Parallel Multiplication:**  
      Same as 7a:
      - Communication: **O(M²/√p₁)**
      - CUDA local multiplication: **O((M/√p₁)³ / cuda_speedup) + overhead**
    - **Sequential Writing:**  
      O(M²)
    - **Overall Time Complexity:**  
      Approximately  
      **O(M²/p₁ + M²/√p₁ + CUDA_multiplication_time)**
      
  - **Speed-up:**  
    Better than 7a due to parallel reading reducing initial I/O bottleneck.

  - **Efficiency:**  
    Improved I/O balance. Efficiency still depends on CUDA kernel performance and MPI communication cost scaling with `p₁`.

  - **Cost:**  
    Same as 7a, with potentially better resource utilization due to reduced read time.

---

Let me know if you'd like this put into a table or diagram for clarity!

# Parallel & Distributed Image Processing

A performance-oriented image processing project implementing and comparing **Serial CPU, OpenMP, MPI, CUDA Naive, and CUDA Shared-Memory** approaches for 3x3 convolution-based image filtering.

The project was developed as an **EIS Semester Project** for the Parallel and Distributed Computing course at HITEC University, Taxila.

## Project Overview

Image convolution is a highly parallel workload because the computation for each output pixel can largely be performed independently.

This project implements three image filters:

* **Gaussian Blur**
* **Sobel Edge Detection**
* **Sharpen**

Each filter is implemented using five different computing approaches:

```text
Serial CPU
    │
    ├── OpenMP
    │
    ├── MPI
    │
    ├── CUDA Naive
    │
    └── CUDA Shared Memory
```

The implementations are benchmarked across image resolutions ranging from **256×256 to 4096×4096 pixels** to study execution time, scalability, and speedup.

## Implementations

### 1. Serial

The serial implementation provides the baseline performance.

It processes the image using standard nested C++ loops and applies the 3x3 convolution kernel sequentially.

```text
1 pixel → processed at a time
```

### 2. OpenMP

OpenMP is used to parallelize the image-processing loops across CPU threads.

The implementation uses:

```cpp
#pragma omp parallel for collapse(2) schedule(static)
```

This allows multiple pixels to be processed concurrently using shared-memory CPU parallelism.

### 3. MPI

The MPI implementation uses distributed-memory parallelism.

The image is divided into horizontal row blocks and distributed among MPI processes.

```text
              Image
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
      Rank 0   Rank 1   Rank 2 ...
        │       │       │
        └── Halo Exchange ──┘
                │
              Result
```

Because a 3x3 convolution requires neighboring rows, adjacent MPI processes exchange boundary rows using **halo exchange**.

The implementation uses:

* `MPI_Scatterv`
* `MPI_Isend`
* `MPI_Irecv`
* `MPI_Waitall`
* `MPI_Gatherv`
* `MPI_Reduce`

### 4. CUDA Naive

The naive CUDA implementation assigns one GPU thread to each output pixel.

```text
1 CUDA thread → 1 pixel
```

Each thread reads the required neighboring pixels directly from global memory and performs the convolution.

The filter kernels are stored in CUDA constant memory for efficient access.

### 5. CUDA Shared Memory

The optimized CUDA implementation uses **shared-memory tiling**.

A 16×16 thread block cooperatively loads an 18×18 tile into shared memory, including the one-pixel halo required for a 3x3 convolution.

```text
16 × 16 output pixels
        +
1-pixel halo
        ↓
18 × 18 shared-memory tile
```

This allows neighboring threads to reuse pixel data and reduces repeated global-memory accesses.

## Image Filters

### Gaussian Blur

Used to smooth the image and reduce noise.

### Sobel Edge Detection

Uses horizontal and vertical Sobel kernels to calculate image gradients and highlight edges.

### Sharpen

Enhances image details and edge contrast.

## Project Architecture

The system uses a common binary image format so that all implementations can process the same input.

```text
Input Image
     │
     ▼
Grayscale Conversion
     │
     ▼
image.bin
     │
     ├──────────────► Serial
     ├──────────────► OpenMP
     ├──────────────► MPI
     ├──────────────► CUDA Naive
     └──────────────► CUDA Shared
                         │
                         ▼
                   Output .bin
                         │
                         ▼
                  Python / NumPy
                         │
                         ▼
                  Visualization
```

Image dimensions are stored separately in:

```text
image_dims.txt
```

## Benchmarking

The project benchmarks all five implementations across:

### Image Sizes

```text
256 × 256
512 × 512
1024 × 1024
2048 × 2048
4096 × 4096
```

### Filters

```text
Gaussian Blur
Sobel Edge Detection
Sharpen
```

The benchmark results are stored in:

```text
benchmark_results.csv
```

Execution time and speedup graphs are generated using Python and Matplotlib.

## Speedup

Speedup is calculated as:

```text
Speedup = Serial Execution Time / Parallel Execution Time
```

This provides a direct comparison between the serial baseline and each parallel implementation.

## Hardware & Environment

The project was developed and executed using **Google Colab** with an NVIDIA T4 GPU.

| Component                 | Technology                      |
| ------------------------- | ------------------------------- |
| CPU Processing            | C++                             |
| Shared-Memory Parallelism | OpenMP                          |
| Distributed Computing     | MPI                             |
| GPU Computing             | CUDA                            |
| GPU                       | NVIDIA T4                       |
| Environment               | Google Colab                    |
| Image Processing          | NumPy / Pillow                  |
| Benchmarking              | Python                          |
| Visualization             | Matplotlib                      |
| GPU Profiling             | Nsight Systems / Nsight Compute |

## Profiling

CUDA implementations were profiled using:

* NVIDIA Nsight Systems
* NVIDIA Nsight Compute

Profiling was used to examine:

* Kernel execution time
* Memory throughput
* GPU occupancy
* Global-memory access behavior
* Performance differences between naive and shared-memory kernels

## Running the Project

The easiest way to run the complete project is through the provided Google Colab notebook.

### Requirements

* Google account
* Google Colab
* NVIDIA GPU runtime for CUDA execution
* Project notebook

### Google Colab Setup

1. Open Google Colab.
2. Upload the project notebook.
3. Select:

```text
Runtime → Change runtime type → GPU
```

4. Run the notebook from beginning to end.

The notebook handles the required setup, compilation, execution, benchmarking, and visualization.

## Expected Workflow

```text
1. Verify GPU
       ↓
2. Prepare input image
       ↓
3. Convert image to grayscale
       ↓
4. Generate image.bin
       ↓
5. Compile Serial implementation
       ↓
6. Compile OpenMP implementation
       ↓
7. Compile MPI implementation
       ↓
8. Compile CUDA implementations
       ↓
9. Run filters
       ↓
10. Collect execution times
       ↓
11. Generate benchmark_results.csv
       ↓
12. Generate performance graphs
       ↓
13. Profile CUDA kernels
```

## Results

The project compares execution time and speedup across all five implementations.

At larger image sizes, the parallel implementations provide greater opportunity for performance improvement because the computational workload becomes significantly larger.

The CUDA shared-memory implementation specifically demonstrates how optimizing memory access can improve GPU convolution performance by reusing neighboring pixel data.

> Benchmark values should be taken directly from `benchmark_results.csv` generated by the actual run. Results can vary depending on the available CPU resources, GPU, and cloud environment.

## Key Concepts Demonstrated

* Serial computation
* Shared-memory parallelism
* Distributed-memory parallelism
* GPU programming
* Image convolution
* Domain decomposition
* MPI halo exchange
* CUDA thread organization
* CUDA shared memory
* GPU memory hierarchy
* Performance benchmarking
* Speedup analysis
* GPU profiling
* Scalability

## Project Structure

```text
parallel-distributed-image-processing/
│
├── README.md
│
├── serial/
│   └── serial.cpp
│
├── openmp/
│   └── openmp_filter.cpp
│
├── mpi/
│   └── mpi_filter.cpp
│
├── cuda/
│   ├── cuda_naive.cu
│   └── cuda_shared.cu
│
├── benchmark/
│   ├── benchmark.py
│   └── benchmark_results.csv
│
├── notebook/
│   └── parallel_image_filters_v2.ipynb
│
├── results/
│   ├── benchmark_plot_gaussian.png
│   ├── benchmark_plot_sobel.png
│   └── benchmark_plot_sharpen.png
│
└── screenshots/
    ├── simulation-output.png
    └── gpu-profile.png
```

## Limitations

* Google Colab provides variable cloud resources, so execution times can vary between runs.
* MPI performance in Colab is limited by the available CPU resources.
* CUDA execution requires an NVIDIA GPU.
* GPU kernel timing and total application timing represent different measurements.
* MPI communication introduces overhead that can become significant for smaller workloads.

## Future Improvements

* Run MPI implementation across multiple physical nodes.
* Implement additional convolution filters.
* Explore larger CUDA block configurations.
* Experiment with additional shared-memory optimization techniques.
* Add CUDA streams for overlapping computation and data transfers.
* Perform more extensive GPU profiling.
* Compare performance across different NVIDIA GPU architectures.

## Authors

**Muhammad Huzaifa** — 23-CS-007
**Syed Ali Tariq** — 23-CS-072
**Muhammad Owais** — 23-CS-055

**Section:** C
**Department of Computer Science**
**HITEC University, Taxila

## Course

**Parallel and Distributed Computing**

**Project Type:** EIS Semester Project

---

*Developed for educational purposes as part of a Parallel and Distributed Computing project.*

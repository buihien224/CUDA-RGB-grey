/*
	kernel.cu
	Holds the kernel for the main program
*/

#include <iostream>

#define BLOCK_WIDTH 32
#define cuda_check_errors(val) check( (val), #val, __FILE__, __LINE__)

using namespace std;

template<typename T>
void check(T err, const char* const func, const char* const file, 
			const int line) {
	if (err != cudaSuccess) {
		cerr << "CUDA error at: " << file << ":" << line << endl;
		cerr << cudaGetErrorString(err) << " " << func << endl;
		exit(1);
	}
}

__global__
void rgba_to_grey(uchar4 *const d_rgba, unsigned char *const d_grey, 
					size_t rows, size_t cols) {
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) 
		return;
	
	uchar4 p = d_rgba[i * cols + j];
	d_grey[i * cols + j] = (unsigned char) (0.299f * p.x + 0.587f * p.y + 0.114f * p.z);
}

void rgba_to_grey_launcher(uchar4 *const d_rgba, unsigned char *const d_grey,
							size_t rows, size_t cols) {
    const dim3 block_size (BLOCK_WIDTH, BLOCK_WIDTH, 1);
    unsigned int grid_x = (unsigned int) (rows / BLOCK_WIDTH + 1);
    unsigned int grid_y = (unsigned int) (cols / BLOCK_WIDTH + 1);
    const dim3 grid_size (grid_x, grid_y, 1);
    rgba_to_grey<<<grid_size, block_size>>>(d_rgba, d_grey, rows, cols);
    cudaDeviceSynchronize();
    cuda_check_errors(cudaGetLastError());
}

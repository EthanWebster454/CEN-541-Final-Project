#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <ctype.h>
#include <cuda.h>
#include <time.h>

#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))
#define	CEIL(a,b)				((a+b-1)/b)

typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;


typedef struct{
	ui i;
	ui j;
}pixelCoords;

// buffers for images
uch *TheImg, *CopyImg;				
uch *GPUImg, *GPUCopyImg, *GPUptr, *GPUResult, *NoiseMap, *KernelIndices;

// noisy pixel locations
pixelCoords *NoisyPixelCoords;

// mutex variables for tracking noisy pixels
ui *GlobalMax, *GlobalMin, *NumNoisyPixelsGPU, *GPUmutexes;


#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)



// Kernel that locates potentially noisy pixels in an image by using impulse noise detection
__global__
void findNoisyPixels(pixelCoords *locations, uch *ImgSrc, uch *noiseMap, ui*globalMax, ui*globalMin, ui*ListLength, ui Hpixels, ui Vpixels)
{

	// 3x3 matrix of pixels around current pixel
	//uch mat3x3[8]; // 3 x 3 - 1 = 8

	// threads/blocks info and IDs
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;

	// leave buffer frame around image to avoid 8 edge cases for convolutions
	if (MYcol > Hpixels-4 || MYcol < 3 || MYrow > Vpixels-4 || MYrow < 3) return;

	ui MYpixIndex = MYrow * Hpixels + MYcol; // pixel index in B&W image

	uch pIJ = ImgSrc[MYpixIndex];
	uch max = 0;
	uch min = 255;
	uch curr;
	uch nMax;
	uch nMin;
	uch oldMax;
	uch oldMin;
	int row;
	int col;
	int indx;
	
	// find min and max pixel intensities in current window
	for (int i = -1; i <= 1; i++){
		for (int j = -1; j <= 1; j++){

			if(!(j==0 && i==0)){

				row = MYrow + i;
				col = MYcol + j;
				indx = row*Hpixels + col;
				curr = ImgSrc[indx];

				if(curr > max)
					max = curr;
				if(curr < min)
					min = curr;
			}
		}
	}

	// atomically update global max and min pixel intensities
	oldMax = atomicMax(globalMax, (ui)max);
	oldMin = atomicMin(globalMin, (ui)min);

	// if the old max wasn't updated, then max is "salt" noise
	// otherwise, we must assume that 255 is "salt" noise
	if(oldMax == max)
		nMax = max;
	else
		nMax = 255;

	// if the old min wasn't updated, then min is "pepper" noise
	// otherwise, we must assume that 0 is "pepper" noise
	if(oldMin == min)
		nMin = min;
	else
		nMin = 0;
	
	// if the current pixel intensity is equal to min or max,
	// then it is likely s&p noise. Mark as such.
	if(pIJ == nMin || pIJ == nMax){

		int listIndex = atomicAdd(ListLength, (ui)1); 

		locations[listIndex].i = MYrow;
		locations[listIndex].j = MYcol;

		noiseMap[MYpixIndex] = 0;

	}
	
}


__device__
uch Horz[5][5] = {	{ 0, 0,  0,  0,  0 },
					{ 1, 1,  1,  1,  1 },
					{ 1, 1,  0,  1,  1 },
					{ 1, 1,  1,  1,  1 },
					{ 0, 0,  0,  0,  0 } };

__device__
uch Vert[5][5] = {	{ 0, 1,  1,  1,  0 },
					{ 0, 1,  1,  1,  0 },
					{ 0, 1,  0,  1,  0 },
					{ 0, 1,  1,  1,  0 },
					{ 0, 1,  1,  1,  0 } };

__device__
uch mask45[7][7]={	{0, 0, 0, 0, 1, 0, 0},
					{0, 0, 0, 1, 1, 1, 0},
					{0, 0, 1, 1, 1, 1, 1},
					{0, 1, 1, 0, 1, 1, 0},
					{1, 1, 1, 1, 1, 0, 0},
					{0, 1, 1, 1, 0, 0, 0},
					{0, 0, 1, 0, 0, 0, 0}};

__device__
uch mask135[7][7]={	{0, 0, 1, 0, 0, 0, 0},
                  	{0, 1, 1, 1, 0, 0, 0},
                  	{1, 1, 1, 1, 1, 0, 0},
                  	{0, 1, 1, 0, 1, 1, 0},
                  	{0, 0, 1, 1, 1, 1, 1},
                  	{0, 0, 0, 1, 1, 1, 0},
                  	{0, 0, 0, 0, 1, 0, 0}};


// Kernel that adds salt&pepper noise of given probability density to an image
__global__
void determineMasks(pixelCoords *locations, uch *ImgSrc, uch *noiseMap, uch *kernelIndices, ui ListLength, ui Hpixels, ui R) {

	// threads/blocks info and IDs
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	// ensure not out-of-bounds
	if (MYgtid > ListLength) return;

	// masked arrays of those pixels denoted as noise-free
	uch noiseFreeLists[60];
	uch *maskA = noiseFreeLists;
	uch *maskB = maskA+14;
	uch *maskC = maskB+14;
	uch *maskD = maskC+14;
	uch *listLengths = maskD+14;
	uch *currMask;
	uch currListLength;

	// control and tracking variables
	int i, j, row, col, indx, maskAIndx=0, maskBIndx=0, maskCIndx=0, maskDIndx=0, chosenMask;
	float minStdDev=1000000.0, currStdDev, sum = 0.0, mean, standardDeviation = 0.0;

	// obtain current noisy pixel indices
	pixelCoords currCoord = locations[MYgtid];

	ui MYrow = currCoord.i;
	ui MYcol = currCoord.j;

	// iterate through both 5x5 masks to find values of noise-free pixels
	for (i = -2; i <= 2; i++){
		for (j = -2; j <= 2; j++){

			// find current absolute index
			row = MYrow + i;
			col = MYcol + j;
			indx = row*Hpixels + col;

			// if the current pixel is noise-free AND
			if(noiseMap[indx]){

				// if the current 5x5 horizontal mask cell is set to TRUE
				if(Horz[i+2][j+2]) {

					// obtain noise free pixel and add to list
					maskA[maskAIndx] = ImgSrc[indx];
					maskAIndx++;
				}

				// if the current 5x5 vertical mask cell is set to TRUE
				if(Vert[i+2][j+2]) {

					// obtain noise free pixel and add to list
					maskB[maskBIndx] = ImgSrc[indx];
					maskBIndx++;
				}

			}
		}
	}

	// iterate through both 7x7 masks to find values of noise-free pixels
	for (i = -3; i <= 3; i++){
		for ( j = -3; j <= 3; j++){

			// find current absolute index
			row = MYrow + i;
			col = MYcol + j;
			indx = row*Hpixels + col;

			// if the current pixel is noise-free AND
			if(noiseMap[indx]){

				// if the current 7x7 45 degree mask cell is set to TRUE
				if(mask45[i+3][j+3]) {
					// obtain noise free pixel and add to list
					maskC[maskCIndx] = ImgSrc[indx];
					maskCIndx++;
				}

				// if the current 7x7 135 degree mask cell is set to TRUE
				if(mask135[i+3][j+3]) {
					// obtain noise free pixel and add to list
					maskD[maskDIndx] = ImgSrc[indx];
					maskDIndx++;
				}

			}
		}
	}

	// if the amounts of noise free pixels in any of the directional masks is 
	// below threshold R, then we use 3x3 convolution
	// this helps to mitigate promoting false edges
	if(maskAIndx < R || maskBIndx < R || maskCIndx < R || maskDIndx < R)
		chosenMask = 0;

	else {

		// assign list lengths for smoother access
		listLengths[0] = maskAIndx;
		listLengths[1] = maskBIndx;
		listLengths[2] = maskCIndx;
		listLengths[3] = maskDIndx;

		// find the mask index (from 1 to 4) of minimum standard deviation
		for(i=0; i < 4; i++) {

			currListLength = listLengths[i];
			currMask = maskA+(i*14);
			
			// first find mean of array
			for(j = 0; j < currListLength; j++)
			{
				sum += (float)currMask[j];
			}
			mean = sum/currListLength;

			// then find sum of individual deviations
			for(j = 0; j < currListLength; j++)
				standardDeviation += pow((float)currMask[j] - mean, 2);

			// final StdDev is normalized by list length
			currStdDev = standardDeviation / currListLength;

			if(currStdDev < minStdDev) {
				chosenMask = i+1;
				minStdDev = currStdDev;
			}

		}

	}

	// assign the mask index that was chosen
	kernelIndices[MYgtid] = chosenMask;

}


// Kernel that calculates a B&W image from an RGB image
// resulting image has a double type for each pixel position
__global__
void BWKernel(uch *ImgBW, uch *ImgGPU, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	double R, G, B;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYsrcIndex = MYrow * RowBytes + 3 * MYcol;
	ui MYpixIndex = MYrow * Hpixels + MYcol;

	B = (double)ImgGPU[MYsrcIndex];
	G = (double)ImgGPU[MYsrcIndex + 1];
	R = (double)ImgGPU[MYsrcIndex + 2];
	ImgBW[MYpixIndex] = (uch)((R+G+B)/3.0);
}


// Kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
__global__
void PixCopy(uch *ImgDst, uch *ImgSrc, ui FS)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	if (MYgtid > FS) return;				// outside the allocated memory
	ImgDst[MYgtid] = ImgSrc[MYgtid];
}


/*
// helper function that wraps CUDA API calls, reports any error and exits
void chkCUDAErr(cudaError_t error_id)
{
	if (error_id != CUDA_SUCCESS)
	{
		printf("CUDA ERROR :::%\n", cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}
*/


// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate memory to store the 1D image and return its pointer.
uch *ReadBMPlin(char* fn)
{
	static uch *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL){	printf("\n\n%s NOT FOUND\n\n", fn);	exit(EXIT_FAILURE); }

	uch HeaderInfo[54];
	fread(HeaderInfo, sizeof(uch), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width * 3 + 3) & (~3);		ip.Hbytes = RowBytes;
	//save header for re-use
	memcpy(ip.HeaderInfo, HeaderInfo,54);
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn, 
			ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img  = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
}


// Write the 1D linear-memory stored image into file.
void WriteBMPlin(uch *Img, char* fn)
{
	FILE* f = fopen(fn, "wb");
	if (f == NULL){ printf("\n\nFILE CREATION ERROR: %s\n\n", fn); exit(1); }
	//write header
	fwrite(ip.HeaderInfo, sizeof(uch), 54, f);
	//write data
	fwrite(Img, sizeof(uch), IMAGESIZE, f);
	printf("\nOutput File name: %17s  (%u x %u)   File Size=%u", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
	fclose(f);
}


int main(int argc, char **argv)
{

	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t cudaStatus;
	cudaEvent_t time1, time2, time3, time4;
	char InputFileName[255], OutputFileName[255], ProgName[255];
	ui BlkPerRow, ThrPerBlk=256, NumBlocks, GPUDataTransfer;
	cudaDeviceProp GPUprop;
	ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100]; 

	ui GPUtotalBufferSize, R = 5, T = 5, NumNoisyPixelsCPU, mutexInit[3] = {0, 255, 0};

	strcpy(ProgName, "randNoiseRemoval");
	switch (argc){
	case 6:  ThrPerBlk = atoi(argv[5]);
	case 5:  R = atoi(argv[4]);
	case 4:  T = atoi(argv[3]);
	case 3:  strcpy(InputFileName, argv[1]);
			 strcpy(OutputFileName, argv[2]);
			 break;
	default: printf("\n\nUsage:   %s InputFilename OutputFilename [T] [R] [ThrPerBlk]", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp 5", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp 5 5",ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp 5 5 128",ProgName);
			 printf("\n\nT = reconstruction threshold, R = mask selection threshold\n\n");
			 exit(EXIT_FAILURE);
	}

	if ((ThrPerBlk < 32) || (ThrPerBlk > 1024)) {
		printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
		exit(EXIT_FAILURE);
	}
	

	// Create CPU memory to store the input and output images
	TheImg = ReadBMPlin(InputFileName); // Read the input image if memory can be allocated
	if (TheImg == NULL){
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	CopyImg = (uch *)malloc(IMAGESIZE);
	if (CopyImg == NULL){
		free(TheImg);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}


	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}

	
	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
	MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;

	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer

/*
	>>> GPU STORAGE DETAILS >>>
	GPUImage: IMAGESIZE
	GPUCopyImage(BW) : IMAGEPIX
	NoisyPixelCoords: IMAGEPIX*sizeof(pixelCoords)
	NoiseMap : IMAGEPIX
	KernelIndices : IMAGEPIX
	GlobalMax : sizeof(ui)
	GlobalMin : sizeof(ui)
	NumNoisyPixelsGPU : sizeof(ui)
*/
	// allocate sufficient memory on the GPU to hold all above items
	GPUtotalBufferSize = IMAGESIZE+(IMAGEPIX*sizeof(pixelCoords))+IMAGEPIX*3+sizeof(ui)*3;
	cudaStatus = cudaMalloc((void**)&GPUptr, GPUtotalBufferSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory for buffers");
		exit(EXIT_FAILURE);
	}

	// setup buffer pointers for functions
	GPUImg			= (uch *)GPUptr;
	GPUCopyImg	= GPUImg + IMAGESIZE;
	NoiseMap = GPUCopyImg + IMAGEPIX;  // add the previous image/array of noisy pixel intensities
	KernelIndices = NoiseMap + IMAGEPIX;
	NoisyPixelCoords = (pixelCoords*)(KernelIndices + IMAGEPIX);
	GlobalMax = (ui*)(NoisyPixelCoords + IMAGEPIX);
	GlobalMin = GlobalMax+1;
	NumNoisyPixelsGPU = GlobalMin+1;

	
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for input image CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	// Copy mutex initializations from CPU to GPU
	cudaStatus = cudaMemcpy(GlobalMax, mutexInit, 3*sizeof(ui), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for mutex initializations CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	// assume pixels are not noisy by default
	cudaStatus = cudaMemset (NoiseMap, 1, IMAGEPIX );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset for Noise Map failed!");
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done

	
	BlkPerRow = CEIL(ip.Hpixels, ThrPerBlk);
	NumBlocks = IPV*BlkPerRow;

	BWKernel <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n cudaDeviceSynchronize for B&WKernel returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	findNoisyPixels <<< NumBlocks, ThrPerBlk >>> (NoisyPixelCoords, GPUCopyImg, NoiseMap, GlobalMax, GlobalMin, NumNoisyPixelsGPU, IPH, IPV);
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize for findNoisyPixels returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time3, 0);

	cudaStatus = cudaMemcpy(&NumNoisyPixelsCPU, NumNoisyPixelsGPU, sizeof(ui), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}


	// only schedule as many threads are needed for NoisyPixelListLength
	NumBlocks = CEIL(NumNoisyPixelsCPU, ThrPerBlk);
	
	determineMasks <<< NumBlocks, ThrPerBlk >>> (NoisyPixelCoords, GPUCopyImg, NoiseMap, KernelIndices, NumNoisyPixelsCPU, IPH, R);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize for determineMasks returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	
	printf("\n*****************************\n");
	printf("The number of noisy pixels found was %d\n", NumNoisyPixelsCPU);
	printf("*****************************\n");


	GPUDataTransfer = GPUtotalBufferSize;

	// Copy output (results) from GPU buffer to host (CPU) memory.
	// cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
	// 	exit(EXIT_FAILURE);
	// }
	cudaEventRecord(time4, 0);

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	//WriteBMPlin(CopyImg, OutputFileName);		// Write the flipped image back to disk
	printf("\n\n--------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n", 
			GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------\n");
	printf("%s %s %s %d %d %u   [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName,
			T, R, ThrPerBlk, NumBlocks, BlkPerRow);
	printf("--------------------------------------------------------------------------\n");
	printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
	printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecutionTime, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));
	printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU));
	printf("--------------------------------------------------------------------------\n");
	printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2 * IMAGESIZE + GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
	printf("--------------------------------------------------------------------------\n\n");

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUptr);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}

	free(TheImg);
	free(CopyImg);
	return(EXIT_SUCCESS);
}




// Salt and pepper noise simulation with Cuda C/C++
// Original framework for code taken from imflipG.cu
// Modified by Ethan Webster

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

uch *TheImg, *CopyImg;					// Where images are stored in CPU
uch *GPUImg, *GPUCopyImg, *GPUptr, *GPUResult;	// Where images are stored in GPU


struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;


#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)


// Kernel that adds salt&pepper noise of given probability density to an image
__global__
void corruptPixels(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, double prob)
{
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

	ui RowBytes = (Hpixels * 3 + 3) & (~3);  // bytes in row of R=G=B grayscale output image

	ui MYresultIndex = MYrow * RowBytes + 3 * MYcol; // pixel index in grayscale image (R=B=G)

	// seed cuRAND random number generator function with clock cycle + threadID
	curandState state;
	curand_init((unsigned long long)clock() + MYtid, 0, 0, &state);

	// sample uniform distribution from 0 to 255 (random pixel intensity)
	ui loc = ((ui)(curand(&state)))%255;

/* 
	half of the probability is used for the following because
	the salt vs pepper contribution is split 50/50 
*/
	// if pixel intensity is located in the lower half of the 
	// probability region, then add pepper noise
	if( loc <= (ui)(prob/2.0f)) {
		ImgDst[MYresultIndex] = 0;
		ImgDst[MYresultIndex+1] = 0;
		ImgDst[MYresultIndex+2] = 0;
	}

	// otherwise if pixel intensity is located in the upper half of the 
	// probability region, then add salt noise
	else if(loc > (ui)(prob/2.0f) && loc < (ui)prob ) {
		ImgDst[MYresultIndex] = 255;
		ImgDst[MYresultIndex+1] = 255;
		ImgDst[MYresultIndex+2] = 255;
	}

	// if we reached this, then no noise is added
	else {
		ImgDst[MYresultIndex] = ImgSrc[MYpixIndex];
		ImgDst[MYresultIndex+1] = ImgSrc[MYpixIndex];
		ImgDst[MYresultIndex+2] = ImgSrc[MYpixIndex];
	}


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

	cudaError_t cudaStatus;
	char InputFileName[255], OutputFileName[255], ProgName[255];
	ui BlkPerRow, ThrPerBlk=256, NumBlocks;
	//cudaDeviceProp GPUprop;
	//ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100]; 

	ui amt, GPUtotalBufferSize;

	double inputNL;

	strcpy(ProgName, "randNoise");
	switch (argc){
	case 5:  ThrPerBlk=atoi(argv[4]);
	case 4:  amt=atoi(argv[3]);
	case 3:  strcpy(InputFileName, argv[1]);
			 strcpy(OutputFileName, argv[2]);
			 break;
	default: printf("\n\nUsage:   %s InputFilename OutputFilename [NoiseDensity] [ThrPerBlk]", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp 50", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp 50  128",ProgName);
			 printf("\n\nNoise Density is in percent, from 0-100\n\n");
			 exit(EXIT_FAILURE);
	}
	if (amt > 100) {
		printf("Invalid noise amount. Must be between 0 and 100");
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

	
	// cudaGetDeviceProperties(&GPUprop, 0);
	// SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
	// SupportedMBlocks = SupportedKBlocks / 1024;
	// sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
	// MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;

	// allocate sufficient memory on the GPU to hold B&W image and grayscale output image
	GPUtotalBufferSize = IMAGEPIX+IMAGESIZE;
	cudaStatus = cudaMalloc((void**)&GPUptr, GPUtotalBufferSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}

	// setup pointers to B&W image and output corrupted image
	GPUImg			= (uch *)GPUptr;
	GPUCopyImg	= GPUImg + IMAGESIZE;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	BlkPerRow = CEIL(ip.Hpixels, ThrPerBlk);
	NumBlocks = IPV*BlkPerRow;

	BWKernel <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n cudaDeviceSynchronize 1 returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	// add random noise to the image
	inputNL = 255.0f*(double)amt/100.0f;
		 corruptPixels <<< NumBlocks, ThrPerBlk >>> (GPUImg, GPUCopyImg, IPH, IPV, inputNL);
				  GPUResult = GPUImg;


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize 2 returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	

	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}
	

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}


	WriteBMPlin(CopyImg, OutputFileName);		// Write the flipped image back to disk

	printf("\n\n--------------------------------------------------------------------------\n");
	printf("Successfully added %d%% noise to the given image and converted to grayscale.\n", amt);

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUptr);

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




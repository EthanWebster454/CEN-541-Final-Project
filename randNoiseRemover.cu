// Salt and pepper noise removal via inpainting with Cuda C/C++
// Original framework for code taken from imflipG.cu
// Modified by Ethan Webster and Ashley Suchy


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

//image properties struct
struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;

// noisy pixel location struct (raster coordinates)
typedef struct{
	ui i; // row
	ui j; // column
}pixelCoords;

// buffers for images, noisy pixel matrix, and kernel indices
uch *TheImg, *CopyImg;				
uch *GPUImg, *GPUCopyImg, *GPUptr, *GPUResult, *NoiseMap, *KernelIndices;
double *GPU_PREV_BW, *GPU_CURR_BW;

// noisy pixel locations
pixelCoords *NoisyPixelCoords;

// mutex variables for tracking noisy pixels and SAD
ui *GlobalMax, *GlobalMin, *NumNoisyPixelsGPU, *GPUmutexes, *GPU_SAD;


#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)


// Kernel that locates potentially noisy pixels in an image by using impulse noise detection
__global__
void findNoisyPixels(pixelCoords *locations, uch *ImgSrc, uch *noiseMap, ui*globalMax, ui*globalMin, ui*ListLength, ui Hpixels, ui Vpixels)
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
	// (this is a PERFECT example of avoiding unecessary complexity!!!)
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



//3x3 standard mask
__constant__
double mask0[3][3] = {  {0.1036,	0.1464,	0.1036},
						{0.1464,	0,		0.1464},
						{0.1036,	0.1464,	0.1036}};

// horizontal 5x5 mask
__constant__
double mask1[5][5] = {  {0,			0,		0,			0,		0		},
						{0.0465,	0.0735,	0.1040,		0.0735,	0.0465	},
						{0.0520,	0.1040,	0,			0.1040,	0.0520	},
						{0.0465,	0.0735,	0.1040,		0.0735,	0.0465	},
						{0,			0,		0,			0,		0		}};

//vertical 5x5 mask						
__constant__
double mask2[5][5] = {  {0,	0.0465,	0.0520,	0.0465,	0},
						{0,	0.0735,	0.1040,	0.0735,	0},
						{0,	0.1040,	0,		0.1040,	0},
						{0,	0.0735,	0.1040,	0.0735,	0},
						{0,	0.0465,	0.0520,	0.0465,	0}};

//45 degree 7x7 mask					
__constant__
double mask3[7][7] = {	{0,			0,		0,		0,		0.0251,	0,		0		},
						{0,			0,		0,		0.0397,	0.0355,	0.0281,	0		},
						{0,			0,		0.0562,	0.0794,	0.0562,	0.0355,	0.0251	},
						{0,			0.0397,	0.0794,	0,		0.0794,	0.0397,	0		},
						{0.0251,	0.0355,	0.0562,	0.0794,	0.0562,	0,		0		},
						{0,			0.0281,	0.0355,	0.0397,	0,		0,		0		},
						{0,			0,		0.0251,	0,		0,		0,		0		}};
						
//135 degree 7x7 mask							
__constant__						
double mask4[7][7] = {  {0,			0,		0.0251,	0,		0,		0,		0		},
						{0,			0.0281,	0.0355,	0.0397,	0,		0,		0		},
						{0.0251,	0.0355,	0.0562,	0.0794,	0.0562,	0,		0		},
						{0,			0.0397,	0.0794,	0,		0.0794,	0.0397,	0		},
						{0,			0,		0.0562,	0.0794,	0.0562,	0.0355,	0.0251	},
						{0,			0,		0,		0.0397,	0.0355,	0.0281,	0		},
						{0,			0,		0,		0,		0.0251,	0,		0		}};


// Kernel that determines appropriate inpainting mask to use for each noisy pixel based on surrounding noiseless pixels
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
	int i, j, row, col, indx, maskAIndx=0, maskBIndx=0, maskCIndx=0, maskDIndx=0, chosenMask=0;
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

				// if the current 5x5 horizontal mask cell is not 0
				if((int)mask1[i+2][j+2]) {

					// obtain noise free pixel and add to list
					maskA[maskAIndx] = ImgSrc[indx];
					maskAIndx++;
				}

				// if the current 5x5 vertical mask cell is not 0
				if((int)mask2[i+2][j+2]) {

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

				// if the current 7x7 45 degree mask cell is not 0
				if((int)mask3[i+3][j+3]) {
					// obtain noise free pixel and add to list
					maskC[maskCIndx] = ImgSrc[indx];
					maskCIndx++;
				}

				// if the current 7x7 135 degree mask cell is not 0
				if((int)mask4[i+3][j+3]) {
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

			// final StdDev^2 is normalized by list length
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




// inpainting convolutions based on kernel indices
__global__
void Convolute(double *ImgCurr, double *ImgBW,  pixelCoords *pc,  uch *kernalI, ui numNoisy, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	if (MYgtid >= numNoisy) return;			// index out of range

	// current noisy pixel coordinates
	ui i=pc[MYgtid].i,j=pc[MYgtid].j,m=kernalI[MYgtid];

	// absolute pixel index
	ui MYpixIndex = i * Hpixels + j;

	int a,b,row,col,index;
	double C = 0.0;

	// based on the kernel index, convolutes with the correct mask
	switch(m)
	{
		case 0: for (a = -1; a <= 1; a++){
					for (b = -1; b <= 1; b++){
						row = i + a;
						col = j + b;
						index = row*Hpixels + col;
						C += (ImgBW[index] * mask0[a + 1][b + 1]);
					}
				}
				ImgCurr[MYpixIndex] = C;
				break;
		case 1: for (a = -2; a <= 2; a++){
					for (b = -2; b <= 2; b++){
						row = i + a;
						col = j + b;
						index = row*Hpixels + col;
						C += (ImgBW[index] * mask1[a + 2][b + 2]);
					}
				}
				ImgCurr[MYpixIndex] = C;
				break;
		case 2: for (a = -2; a <= 2; a++){
					for (b = -2; b <= 2; b++){
						row = i + a;
						col = j + b;
						index = row*Hpixels + col;
						C += (ImgBW[index] * mask2[a + 2][b + 2]);
					}
				}
				ImgCurr[MYpixIndex] = C;
				break;
		case 3: for (a = -3; a <= 3; a++){
					for (b = -3; b <= 3; b++){
						row = i + a;
						col = j + b;
						index = row*Hpixels + col;
						C += (ImgBW[index] * mask3[a + 3][b + 3]);
					}
				}
				ImgCurr[MYpixIndex] = C;
				break;
		default: for (a = -3; a <= 3; a++){
					for (b = -3; b <= 3; b++){
						row = i + a;
						col = j + b;
						index = row*Hpixels + col;
						C += (ImgBW[index] * mask4[a + 3][b + 3]);
					}
				}

				// assign convolution sum to current noisy pixel index
				ImgCurr[MYpixIndex] = C;
				break;
	}
		
}


// sum of absolute differences, reconstruction progress tracking mechanism
__global__
void SAD(ui *sad, double *prev, double *current, pixelCoords *pc, ui numNoisy, ui Hpixels, ui Vpixels)
{
	// thread IDs
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	
	if (MYgtid >= numNoisy) return;	// index out of range

	ui i=pc[MYgtid].i, j=pc[MYgtid].j; // current noisy pixel coordinates

	ui MYpixIndex = i * Hpixels + j; // absolute index
	
	// difference of old and updated pixel values, round to nearest integer
	int absDiff=(int)(prev[MYpixIndex]-current[MYpixIndex]+0.5);

	// absolute difference
	if(absDiff<0)
		absDiff = -absDiff;
	
	atomicAdd(sad, (ui)absDiff); // update global sum

}


// Kernel that calculates a B&W image from an RGB image
// resulting image has a double type for each pixel position
// and a uch type for noisy pixel tracking
__global__
void BWKernel(uch *ImgBW, uch *ImgGPU, double *ImgfpBW, ui Hpixels)
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
	ImgfpBW[MYpixIndex] = (R+G+B)/3.0;
}


// Kernel that calculates a RGB (grayscale) version of B&W image for filing as Windows BMP
__global__
void RGBKernel(uch *ImgRGB, double *ImgBW, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYdstIndex = MYrow * RowBytes + 3 * MYcol;
	ui MYpixIndex = MYrow * Hpixels + MYcol;

	uch pixInt = ImgBW[MYpixIndex];

	// trivial copying: copy R=G=B = B&W pixel intensity
	ImgRGB[MYdstIndex] = pixInt;
	ImgRGB[MYdstIndex+1] = pixInt;
	ImgRGB[MYdstIndex+2] = pixInt;
}


// Kernel that copies just the noisy pixels from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
__global__
void NoisyPixCopy(double *NPDst, double *ImgSrc, pixelCoords *pc, ui NoisyPixelListLength, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	if (MYgtid >= NoisyPixelListLength) return;// outside the allocated memory

	pixelCoords currCoord = pc[MYgtid];

	ui srcIndex = currCoord.i * Hpixels + currCoord.j;

	NPDst[srcIndex] = ImgSrc[srcIndex];
}


// Kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
__global__
void PixCopy(double *ImgDst, double *ImgSrc, ui FS)
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

	float /*totalTime, tfrCPUtoGPU, tfrGPUtoCPU,*/ kernelExecutionTime; // GPU code run times
	cudaError_t cudaStatus;
	cudaEvent_t time1, time2;//, time3, time4;
	char InputFileName[255], OutputFileName[255], ProgName[255];
	ui BlkPerRow, ThrPerBlk=256, NumBlocks, /* GPUDataTransfer,*/ NumBlocksNP;
	cudaDeviceProp GPUprop;
	ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100]; 

	ui GPUtotalBufferSize, R = 5, T = 5, NumNoisyPixelsCPU, mutexInit[4] = {0, 255, 0, 0};
	ui CPU_SAD;

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

	

/*
	>>> GPU STORAGE DETAILS >>>
	***********************
	GPUImage: IMAGESIZE
	GPUCopyImage(BW) : IMAGEPIX
	NoisyPixelCoords: IMAGEPIX*sizeof(pixelCoords)
	NoiseMap : IMAGEPIX
	KernelIndices : IMAGEPIX
	GlobalMax : sizeof(ui)
	GlobalMin : sizeof(ui)
	NumNoisyPixelsGPU : sizeof(ui)
	GPU_PREV_BW : sizeof(double) * IMAGEPIX
	GPU_CURR_BW : sizeof(double) * IMAGEPIX  
	GPU_SAD : sizeof(ui)
	***********************

*/
	// allocate sufficient memory on the GPU to hold all above items
	GPUtotalBufferSize = IMAGESIZE+(IMAGEPIX*sizeof(pixelCoords))+IMAGEPIX*3+sizeof(ui)*4+2*(sizeof(double)*IMAGEPIX);
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
	GPU_PREV_BW = (double*)(NoisyPixelCoords+IMAGEPIX);
	GPU_CURR_BW = GPU_PREV_BW + IMAGEPIX;
	GlobalMax = (ui*)(GPU_CURR_BW + IMAGEPIX);
	GlobalMin = GlobalMax+1;
	NumNoisyPixelsGPU = GlobalMin+1;
	GPU_SAD = NumNoisyPixelsGPU+1;


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for input image CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	// Copy mutex initializations from CPU to GPU
	cudaStatus = cudaMemcpy(GlobalMax, mutexInit, 4*sizeof(ui), cudaMemcpyHostToDevice);
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

	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer

	// calculate GPU-specific parameters
	BlkPerRow = CEIL(ip.Hpixels, ThrPerBlk);
	NumBlocks = IPV*BlkPerRow;
	
	// transform RGB input image into grayscale
	BWKernel <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, GPU_CURR_BW, IPH);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\n cudaDeviceSynchronize for B&WKernel returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	// call kernel to locate the noisy pixels in the image
	findNoisyPixels <<< NumBlocks, ThrPerBlk >>> (NoisyPixelCoords, GPUCopyImg, NoiseMap, GlobalMax, GlobalMin, NumNoisyPixelsGPU, IPH, IPV);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize for findNoisyPixels returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}


	// copy the length of the list holding the noisy pixel locations from the GPU to CPU
	cudaStatus = cudaMemcpy(&NumNoisyPixelsCPU, NumNoisyPixelsGPU, sizeof(ui), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy of NumNoisyPixels, GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}


	// only schedule as many threads are needed for NumNoisyPixelsCPU
	NumBlocksNP = CEIL(NumNoisyPixelsCPU, ThrPerBlk);
	
	// determineMasks tries to find the optimal inpainting masks to use for each noisy pixel
	determineMasks <<< NumBlocksNP, ThrPerBlk >>> (NoisyPixelCoords, GPUCopyImg, NoiseMap, KernelIndices, NumNoisyPixelsCPU, IPH, R);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize for determineMasks returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	// intially copy the current working version of the image to gain a previous version
	PixCopy <<< NumBlocks, ThrPerBlk >>> (GPU_PREV_BW, GPU_CURR_BW, IMAGEPIX);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize for PixCopy returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}


	// progress tracking 
	do{

		// reset SAD (sum of absolute pixel differences)
		cudaStatus = cudaMemset (GPU_SAD, 0, sizeof(ui) );
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset for GPU_SAD failed!");
			exit(EXIT_FAILURE);
		}
		
		// perform convolutions with appropriate inpainting masks
		Convolute <<< NumBlocksNP, ThrPerBlk >>> (GPU_CURR_BW, GPU_PREV_BW,  NoisyPixelCoords,  KernelIndices, NumNoisyPixelsCPU, IPH);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n\n cudaDeviceSynchronize for Convolute returned error code %d after launching the kernel!\n", cudaStatus);
			exit(EXIT_FAILURE);
		}

		// find sum of absolute differences for just the pixels denoted as noisy
		SAD <<< NumBlocksNP, ThrPerBlk >>> (GPU_SAD, GPU_PREV_BW, GPU_CURR_BW, NoisyPixelCoords, NumNoisyPixelsCPU, IPH, IPV);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n\n cudaDeviceSynchronize for SAD returned error code %d after launching the kernel!\n", cudaStatus);
			exit(EXIT_FAILURE);
		}

		// copy just the noisy pixel intensities from the current working image version to the previous version
		NoisyPixCopy <<< NumBlocksNP, ThrPerBlk >>> (GPU_PREV_BW, GPU_CURR_BW, NoisyPixelCoords, NumNoisyPixelsCPU, IPH);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n\n cudaDeviceSynchronize for NoisyPixCopy returned error code %d after launching the kernel!\n", cudaStatus);
			exit(EXIT_FAILURE);
		}

		// CudaMemcpy the SAD from GPU to CPU (it is a GPU variable)
		cudaStatus = cudaMemcpy(&CPU_SAD, GPU_SAD, sizeof(ui), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy of SAD from GPU to CPU  failed!");
			exit(EXIT_FAILURE);
		}


	} while(CPU_SAD > T); // iterate until the sum of absolute differences is below threshold


	// must convert floating point B&W back to unsigned char format
	NumBlocks = IPV*BlkPerRow;
	RGBKernel <<< NumBlocks, ThrPerBlk >>> (GPUImg, GPU_CURR_BW, IPH);
	GPUResult = GPUImg;

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done


	//Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);

	cudaEventElapsedTime(&kernelExecutionTime, time1, time2);


	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}

	WriteBMPlin(CopyImg, OutputFileName);		// Write the denoised image back to disk

	printf("\n\n--------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n", 
			GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------\n");
	printf("%s %s %s %d %d %u   [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName,
			T, R, ThrPerBlk, NumBlocks, BlkPerRow);
	
	printf("Kernel Execution    =%7.2f ms\n", kernelExecutionTime);
	printf("--------------------------------------------------------------------------\n\n");

	// Deallocate CPU, GPU memory and destroy events
	cudaFree(GPUptr);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);

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




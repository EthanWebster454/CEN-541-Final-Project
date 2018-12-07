void Convolute(double *imgCurr,double *imgBW,unsigned int Hpixels,pixelCoords *pc,unsigned int *kernalI)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	ui i=pc[MYgtid].i,j=pc[MYgtid],m=kernalI[MYgtid];
	ui MYpixIndex = i * Hpixels + j;
	int a,row,col,index;
	double C = 0.0;
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
				ImgCurr[MYpixIndex] = C;
				break;
	}
		
}
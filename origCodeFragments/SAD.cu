__global__
void SAD(double *sad,double *prev,double *current,unsigned int Hpixels,unsigned int Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	int MYrow = MYbid / BlkPerRow;
	int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	
	absDiff=prev[MYpixIndex]-current[MYpixIndex]; //is this correct?????
	if(absDiff<0)
		absDiff=absDiff*(-1);
	sad[MYpixIndex]=absDiff;
	
	
}
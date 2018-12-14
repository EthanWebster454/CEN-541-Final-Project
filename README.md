# Image Random Noise Simulation and Inpainting
Final project for CEN 541, a graduate course in parallel programming at SUNY Albany, NY

Requires Cuda toolkit
To simulate random noise, compile for Linux by using "make randNoise" then "make randNoiseRemover"

./randNoise [inputImageName.bmp] [outputImageName.bmp] 50
will add 50% salt-and-pepper noise to an image

./randNoiseRemover [inputImageName.bmp] [outputImageName.bmp] [T] [R] [ThrPerBlock]
will reconstruct image by using inpainting. 
T is the reconstruction threshold (lower means better result, min=0) -- optional
R is an inpainting mask selection threshold (set to 5, usually) -- optional
ThrPerBlock is the block size for the GPU -- optional

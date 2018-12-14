randNoise	:	randNoise.cu
			/usr/local/cuda/bin/nvcc randNoise.cu -o randNoise

randNoiseRemover	:	randNoiseRemover.cu
				/usr/local/cuda/bin/nvcc randNoiseRemover.cu -o randNoiseRemover
include("src/fft.jl")


A = rand(ComplexF64, 4, 4, 4);
#A = CUDA.rand(ComplexF64, 4, 4, 4);
transforms = (FFT(), FFT(), FFT());
dims = (1, 2, 3);
fft(A, transforms, dims);

#fft(A, transforms, dims);
@time fft(A, transforms, dims);

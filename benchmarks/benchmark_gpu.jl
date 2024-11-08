__precompile__(false)
using Distributed
using Base: time_ns

@everywhere begin
    using Dagger
    using LinearAlgebra
    using FFTW
    using Random
    using CUDA
    using DaggerGPU
end


include("../src/fft.jl")
@everywhere include("../src/fft.jl")


num_gpus = length(CUDA.devices())

scope = Dagger.scope(cuda_gpus=1:4)

A = rand(ComplexF64, 1024, 1024, 1024)
a = distribute(A, Blocks(1024, 1024, 256))
b = distribute(A, Blocks(512, 512, 1024))
#c = distribute(A, Blocks(256, 256, 512))



for iter in 1:5
    start_time = time_ns()
    
    Dagger.with_options(;scope) do
        @time fft(a, b, (FFT(), FFT(), FFT()), (1, 2, 3), decomp = Slab())
    end
    
    elapsed_time = (time_ns() - start_time) / 1e9  # Convert to seconds
    
    if iter == 5
        println("Time taken in last iteration = $elapsed_time seconds")
    end
end
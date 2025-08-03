__precompile__(false)
using Distributed
using Base: time_ns

#if nprocs() < 5
#    addprocs(4)
#end

@everywhere using Pkg
@everywhere Pkg.instantiate()

#@everywhere Pkg.add("Dagger")
#@everywhere Pkg.add("FFTW")
#@everywhere Pkg.add("KernelAbstractions")
#@everywhere Pkg.add("AbstractFFTs")
#@everywhere Pkg.add("CUDA")
#@everywhere Pkg.add("GPUArrays")
@everywhere using Dagger
@everywhere using LinearAlgebra
@everywhere using FFTW
@everywhere using Random
@everywhere using DaggerGPU, GPUArrays, KernelAbstractions, AbstractFFTs




include("../src/fft.jl")
@everywhere include("../src/fft.jl")


#num_gpus = length(CUDA.devices())

GPUArray = CuArray
scope = Dagger.scope(;cuda_gpu=1)

if myid() == 1 
    A = rand(ComplexF64, 256, 256, 256)
    a = distribute(A, Blocks(256, 256, 64))
    b = distribute(A, Blocks(128, 128, 256))
 #   c = distribute(A, Blocks(64, 64, 128))
else
    a = distribute(nothing, Blocks(256, 256, 64))
    b = distribute(nothing, Blocks(128, 128, 256))
 #   c = distribute(nothing, Blocks(64, 64, 128))
end


for iter in 1:10
    start_time = time_ns()
    
    Dagger.with_options(;scope) do
        @time fft(a, b, (FFT(), FFT(), FFT()), (1, 2, 3), decomp = Slab())
    end
    
    elapsed_time = (time_ns() - start_time) / 1e9  # Convert to seconds
    
    if iter == 5
        println("Time taken in last iteration = $elapsed_time seconds")
    end
end
__precompile__(false)
using Distributed
using Base: time_ns
if nprocs() < 5
    addprocs(4)
end

@everywhere begin
    using Dagger
    using LinearAlgebra
    using FFTW
    using Random
end


include("../src/fft.jl")
@everywhere include("../src/fft.jl")


if myid() == 1 
    A = rand(ComplexF64, 64, 64, 64)
    a = distribute(A, Blocks(64, 64, 16))
    b = distribute(A, Blocks(32, 32, 64))
    c = distribute(A, Blocks(32, 32, 64))
else
    a = distribute(nothing, Blocks(64, 64, 16))
    b = distribute(nothing, Blocks(32, 32, 64))
    c = distribute(nothing, Blocks(32, 32, 64))
end


for iter in 1:5
    start_time = time_ns()
   

  #  Dagger.with_options(;scope) do
        @time fft(a, b, c, (FFT(), FFT(), FFT()), (1, 2, 3), decomp = Pencil())
   # end
    
    elapsed_time = (time_ns() - start_time) / 1e9  # Convert to seconds
    
    if iter == 5
        println("Time taken in last iteration = $elapsed_time seconds")
    end
end
__precompile__(false)
using Distributed
using Base: time_ns
if nprocs() < 5
    addprocs(4)
end
#addprocs([("d3095", 2), ("d3096", 2)])
@everywhere using Pkg
@everywhere Pkg.instantiate()

@everywhere using Dagger
@everywhere using LinearAlgebra
@everywhere using FFTW
@everywhere using Random
@everywhere using Statistics
#@everywhere Pkg.add("DaggerGPU")
#@everywhere using DaggerGPU


#include("../src/fft.jl")
@everywhere include("../src/fft.jl")


if myid() == 1 
    A = rand(ComplexF64, 512, 512, 512)
    a = distribute(A, Blocks(512, 256, 256))
    b = distribute(A, Blocks(256, 512, 256))
    c = distribute(A, Blocks(256, 256, 512))
else
    a = distribute(nothing, Blocks(512, 256, 256))
    b = distribute(nothing, Blocks(256, 512, 256))
    c = distribute(nothing, Blocks(256, 256, 512))
end


@everywhere function create_darray(A::AbstractArray{T,N}, blocks::Blocks{N}) where {T,N}
    domain = ArrayDomain(map(Base.OneTo, size(A)))
    
    #calculate subdomain
    dims = size(A)
    block_sizes = blocks.blocksize
    subdomain_sizes = map((d, b) -> [b for _ in 1:ceil(Int, d/b)], dims, block_sizes) #round_up
    subdomain_cumlengths = map(cumsum, subdomain_sizes)
    
    #create subdomains
    subdomains = Array{ArrayDomain{N}, N}(undef, map(length, subdomain_sizes))
    for idx in CartesianIndices(subdomains)
        starts = map((cumlength, i) -> i == 1 ? 1 : cumlength[i-1] + 1, subdomain_cumlengths, idx.I)
        ends = map(getindex, subdomain_cumlengths, idx.I)
        subdomains[idx] = ArrayDomain(map((s, e) -> s:e, starts, ends))
    end
    
    #create chunks
    chunks = Array{Any,N}(undef, size(subdomains))
    for idx in CartesianIndices(chunks)
        subdomain = subdomains[idx]
        view_indices = subdomain.indexes
        chunks[idx] = Dagger.tochunk(view(A, view_indices...))
    end
    
    DArray{T,N,typeof(blocks),typeof(cat)}(
        domain,
        subdomains,
        chunks,
        blocks,
        cat
    )
end



function benchmark_fft(iters=10)
    forward_times = Float64[]
    backward_times = Float64[]
    scope = Dagger.scope(:default)
    for iter in 1:iters
        forward_start = time_ns()
  #      Dagger.with_options(;scope) do
        @time begin
            result_b = fft(a, b, c, (FFT(), FFT(), FFT()), (1, 2, 3), decomp = Pencil())
        end
   #     end
        forward_time = (time_ns() - forward_start) / 1e9
        push!(forward_times, forward_time)
        
        backward_start = time_ns()
   #     Dagger.with_options(;scope) do
        @time begin
            result_c = ifft(c, b, a, (IFFT(), IFFT(), IFFT()), (1, 2, 3), decomp = Pencil())
   #     end
        end
        backward_time = (time_ns() - backward_start) / 1e9
        push!(backward_times, backward_time)
        
        if iter == iters
            println("\nResults for iteration $iter:")
            println("Forward FFT time: $(forward_time) seconds")
            println("Backward FFT time: $(backward_time) seconds")
            println("Total time: $(forward_time + backward_time) seconds")
            
            fwd_times = forward_times[2:end]
            bwd_times = backward_times[2:end]
            println("\nStatistics (excluding first iteration):")
            println("Forward FFT - Mean: $(mean(fwd_times)), Std: $(std(fwd_times))")
            println("Backward FFT - Mean: $(mean(bwd_times)), Std: $(std(bwd_times))")
        end
    end
end

benchmark_fft()
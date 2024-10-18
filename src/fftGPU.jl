# src/fftGPU.jl

#module DaggerFFT
#__precompile__(false)
using Distributed
using KernelAbstractions, AbstractFFTs, LinearAlgebra, FFTW, Dagger, CUDA, CUDA.CUFFT, Random, GPUArrays
using DaggerGPU

const R2R_SUPPORTED_KINDS = (
    FFTW.DHT,
    FFTW.REDFT00,
    FFTW.REDFT01,
    FFTW.REDFT10,
    FFTW.REDFT11,
    FFTW.RODFT00,
    FFTW.RODFT01,
    FFTW.RODFT10,
    FFTW.RODFT11,
)

"""
DHT (Discrete Hartley Transform):
The DHT is its own inverse, so forward and backward are the same.

REDFT (Real Even Discrete Fourier Transform):
REDFT00 (Type I DCT): Its own inverse (symmetric)
REDFT10 (Type II DCT): Forward transform
REDFT01 (Type III DCT): Backward transform (inverse of REDFT10)
REDFT11 (Type IV DCT): Its own inverse (symmetric)

RODFT (Real Odd Discrete Fourier Transform):
RODFT00 (Type I DST): Its own inverse (symmetric)
RODFT10 (Type II DST): Forward transform
RODFT01 (Type III DST): Backward transform (inverse of RODFT10)
RODFT11 (Type IV DST): Its own inverse (symmetric)
"""

struct FFT end
struct RFFT end
struct IRFFT end
struct IFFT end
struct FFT! end
struct RFFT! end
struct IRFFT! end
struct IFFT! end

abstract type Decomposition end

struct Pencil <: Decomposition end
struct Slab <: Decomposition end

export FFT, RFFT, IRFFT, IFFT, FFT!, RFFT!, IRFFT!, IFFT!, fft, ifft, R2R, R2R!

struct R2R{K}
    kind::K
    function R2R(kind::K) where {K}
        if kind ∉ R2R_SUPPORTED_KINDS
            throw(ArgumentError("Unsupported R2R transform kind: $(kind2string(kind))"))
        end
        new{K}(kind)
    end
end

struct R2R!{K}
    kind::K
    function R2R!(kind::K) where {K}
        if kind ∉ R2R_SUPPORTED_KINDS
            throw(ArgumentError("Unsupported R2R transform kind: $(kind2string(kind))"))
        end
        new{K}(kind)
    end
end

function find_factors(N)
    n = Int(floor(sqrt(N)))
    while N % n != 0
        n -= 1
    end
    m = N ÷ n
    return n, m
end

function plan_transform(transform, A, dims; kwargs...)
        if transform isa FFT
            return CUDA.CUFFT.plan_fft(A, dims; kwargs...)
        elseif transform isa IFFT
            return CUDA.CUFFT.plan_ifft(A, dims; kwargs...)
        elseif transform isa FFT!
            return CUDA.CUFFT.plan_fft!(A, dims; kwargs...)
        elseif transform isa IFFT!
            return CUDA.CUFFT.plan_ifft!(A, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
end

function plan_transform(transform, A, dims, n; kwargs...)
        if transform isa RFFT
            return CUDA.CUFFT.plan_rfft(A, dims; kwargs...)
        elseif transform isa IRFFT
            return CUDA.CUFFT.plan_irfft(A, n, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
end


indexes(a::ArrayDomain) = a.indexes

Base.getindex(arr::CuArray, d::ArrayDomain) = arr[indexes(d)...]

Base.getindex(arr::KernelAbstractions.AbstractArray, d::ArrayDomain) = arr[indexes(d)...]


kind(transform::R2R) = transform.kind
kind(transform::R2R!) = transform.kind

function plan_transform(transform::Union{R2R, R2R!}, A, dims; kwargs...)
    kd = kind(transform)
    if transform isa R2R
        return FFTW.plan_r2r(A, kd, dims; kwargs...)
    elseif transform isa R2R!
        return FFTW.plan_r2r!(A, kd, dims; kwargs...)
    else
        throw(ArgumentError("Unknown transform type"))
    end
end


function create_darray(A::AbstractArray{T,N}, blocks::Blocks{N}) where {T,N}
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


function apply_fft!(out_part, a_part, transform, dim)
    plan = plan_transform(transform, a_part, dim)
    if transform isa Union{FFT!, IFFT!}
        out_part .= plan * a_part  # In-place transform
    else
  #      result = plan * a_part  # Out-of-place transform
  #      copyto!(out_part, result)
           out_part .= plan * a_part 
    end
end

function apply_fft!(out_part, a_part, transform, dim, n)
    plan = plan_transform(transform, a_part, dim, n)
    out_part .= plan * a_part 
end


@kernel function transpose_kernel!(dst, src, src_size_x, src_size_y, src_size_z,
    dst_size_x, dst_size_y, dst_size_z,
    src_offset_x, src_offset_y, src_offset_z,
    dst_offset_x, dst_offset_y, dst_offset_z)
    i, j, k = @index(Global, NTuple)

    src_i = i + src_offset_x
    src_j = j + src_offset_y
    src_k = k + src_offset_z

    dst_i = i + dst_offset_x
    dst_j = j + dst_offset_y
    dst_k = k + dst_offset_z

    if src_i <= src_size_x && src_j <= src_size_y && src_k <= src_size_z &&
        dst_i <= dst_size_x && dst_j <= dst_size_y && dst_k <= dst_size_z
        dst[dst_i, dst_j, dst_k] = src[src_i, src_j, src_k]
    end
end

function transpose(src::DArray{T,3}, dst::DArray{T,3}) where T
    for (src_idx, src_chunk) in enumerate(src.chunks)
        src_data = fetch(src_chunk)
        for (dst_idx, dst_chunk) in enumerate(dst.chunks)
          dst_domain = dst.subdomains[dst_idx]
          src_domain = src.subdomains[src_idx]

          intersect_domain = intersect(src_domain, dst_domain)

          if !isempty(intersect_domain)

            src_indices = relative_indices(intersect_domain, src_domain)
            dst_indices = relative_indices(intersect_domain, dst_domain)

            dst_chunk_data = fetch(dst_chunk)

            if size(dst_chunk_data) != size(dst_domain)
                dst_chunk_data = similar(dst_chunk_data, size(dst_domain)...)
            end

            intersect_size = map(r -> length(r), intersect_domain.indexes)

            #offsets
            src_offset = map(r -> r.start - 1, src_indices)
            dst_offset = map(r -> r.start - 1, dst_indices)

            backend = get_backend(src_data)

            kernel = transpose_kernel!(backend)
            kernel(dst_chunk_data, src_data, 
            size(src_data)..., size(dst_chunk_data)...,
            src_offset..., dst_offset..., ndrange=intersect_size)
            KernelAbstractions.synchronize(backend)

            dst.chunks[dst_idx] = Dagger.tochunk(dst_chunk_data)
          end
        end
    end         
end


function relative_indices(sub_domain, full_domain)
    return map(pair -> (pair[1].start:pair[1].stop) .- (pair[2].start - 1), 
    zip(sub_domain.indexes, full_domain.indexes))
end

@kernel function transpose_kernel_2d!(dst, src, src_size_x, src_size_y,
    dst_size_x, dst_size_y,
    src_offset_x, src_offset_y,
    dst_offset_x, dst_offset_y)
    i, j = @index(Global, NTuple)

    src_i = i + src_offset_x
    src_j = j + src_offset_y

    dst_i = i + dst_offset_x
    dst_j = j + dst_offset_y

    if src_i <= src_size_x && src_j <= src_size_y &&
       dst_i <= dst_size_x && dst_j <= dst_size_y
        dst[dst_i, dst_j] = src[src_i, src_j]
    end
end

@everywhere function transpose(src::DArray{T,2}, dst::DArray{T,2}) where T
    for (src_idx, src_chunk) in enumerate(src.chunks)
        src_data = fetch(src_chunk)
        for (dst_idx, dst_chunk) in enumerate(dst.chunks)
            dst_domain = dst.subdomains[dst_idx]
            src_domain = src.subdomains[src_idx]

            intersect_domain = intersect(src_domain, dst_domain)

            if !isempty(intersect_domain)
                src_indices = relative_indices(intersect_domain, src_domain)
                dst_indices = relative_indices(intersect_domain, dst_domain)

                dst_chunk_data = fetch(dst_chunk)

                if size(dst_chunk_data) != size(dst_domain)
                    dst_chunk_data = similar(dst_chunk_data, size(dst_domain)...)
                end

                intersect_size = map(r -> length(r), intersect_domain.indexes)

                # Offsets
                src_offset = map(r -> r.start - 1, src_indices)
                dst_offset = map(r -> r.start - 1, dst_indices)

                backend = get_backend(src_data)

                kernel = transpose_kernel_2d!(backend)
                kernel(dst_chunk_data, src_data, 
                       size(src_data)..., size(dst_chunk_data)...,
                       src_offset..., dst_offset..., ndrange=intersect_size)
                KernelAbstractions.synchronize(backend)

                dst.chunks[dst_idx] = Dagger.tochunk(dst_chunk_data)
            end
        end
    end         
end


function closest_factors(n::Int)
    factors = [(i, div(n, i)) for i in 1:floor(Int, sqrt(n)) if n % i == 0]
    return argmin(abs(x[1] - x[2]) for x in factors)
end

"
user should call the function with:
    Dagger.fft(A, transforms, dims) #default pencil
    Dagger.fft(A, transforms, dims, decomp=Pencil())
    Dagger.fft(A, transforms, dims, decomp=Slab())
    transforms = (FFT(), FFT(), FFT())
or    transforms = (R2R(FFTW.REDFT10), R2R(FFTW.REDFT10), R2R(FFTW.REDFT10))
or    transforms = (RFFT(), FFT(), FFT())
    dims = (1, 2, 3)
"
#out-of-place
function fft(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{FFT,RFFT,R2R}},
    dims::NTuple{N,Int};
    decomp::Union{Pencil,Nothing} = nothing
) where {T,N}
   # backend = get_backend(A)
    #GPUArray = CuArray

    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)
    if N == 1
        x = size(A, 1)
        a = DArray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
        end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa RFFT # R2C
            a = DArray(A, Blocks(x, div(y, m)))
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, 2) + 1, div(y, m)))
            b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, (2*n)) + 1, y))
        elseif T <: Real && all(transform -> transform isa FFT, transforms) # R2C
            a = DArray(A, Blocks(x, div(y, m)))
            buffer = DArray(ComplexF64.(A), Blocks(x, div(y, m)))
            b = DArray(ComplexF64.(A), Blocks(div(x, n), y))
        else # C2C or R2R
            a = DArray(A, Blocks(x, div(y, m)))
            b = DArray(A, Blocks(div(x, n), y))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(y))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose(In(a), Out(b))
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
        end
        end

        return collect(b)

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa RFFT #R2C
            a =  DArray(A, Blocks(x, div(y, n), div(z, m)))
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2) + 1, div(y, n), div(z, m)))
            b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, (2*n)) + 1, y, div(z, m)))
            c = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, (2*n)) + 1, div(y, m), z))
        elseif T <: Real && all(transform -> transform isa FFT, transforms) #R2C
            a =  DArray(A, Blocks(x, div(y, n), div(z, m)))
            buffer = DArray(ComplexF64.(A), Blocks(x, div(y, n), div(z, m)))
            b = DArray(ComplexF64.(A), Blocks(div(x, n), y, div(z, m))) 
            c = DArray(ComplexF64.(A), Blocks(div(x, n), div(y, m), z))
        else #C2C R2R
        a =  DArray(A, Blocks(x, div(y, n), div(z, m)))
        b =  DArray(A, Blocks(div(x, n), y, div(z, m)))
        c =  DArray(A, Blocks(div(x, n), div(y, m), z))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(z))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose(In(a), Out(b))

            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end

            Dagger.@spawn transpose(In(b), Out(c))
        
            for idx in 1:length(c.chunks)
                c_part = c.chunks[idx]
                Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[3]), In(dims[3]))
            end
        end
        end

        return collect(c)
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

function fft(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{FFT,RFFT,R2R}},
    dims::NTuple{N,Int};
    decomp::Slab
) where {T,N}

    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)
    if N == 1
        x = size(A, 1)
        a = DArray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
        end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa RFFT # R2C
            a = DArray(A, Blocks(x, div(y, num_gpus)))
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, 2) + 1, div(y, num_gpus)))
            b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, (2*num_gpus)) + 1, y))
        elseif T <: Real && all(transform -> transform isa FFT, transforms) # R2C
            a = DArray(A, Blocks(x, div(y, num_gpus)))
            buffer = DArray(ComplexF64.(A), Blocks(x, div(y, num_gpus)))
            b = DArray(ComplexF64.(A), Blocks(div(x, num_gpus), y))
        else # C2C or R2R
            a = DArray(A, Blocks(x, div(y, num_gpus)))
            b = DArray(A, Blocks(div(x, num_gpus), y))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(y))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose(In(a), Out(b))
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
        end
        end

        return collect(b)

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa RFFT #R2C
            a =  DArray(A, Blocks(x, y, div(z, num_gpus)))
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2) + 1, y, div(z, num_gpus)))
            b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, (2*n)) + 1, div(y, m), z))
        elseif T <: Real && all(transform -> transform isa FFT, transforms) #R2C
            a =  DArray(A, Blocks(x, y, div(z, num_gpus)))
            buffer = DArray(ComplexF64.(A), Blocks(x, y, div(z, num_gpus)))
            b = DArray(ComplexF64.(A), Blocks(div(x, n), div(y, m), z)) 
        else #C2C
            a =  DArray(A, Blocks(x, y, div(z, num_gpus)))
            b =  DArray(A, Blocks(div(x, n), div(y, m), z))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), (dims[1], dims[2]), In(z))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), (dims[1], dims[2]))
                end
                Dagger.@spawn transpose(In(buffer), Out(b))
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), (dims[1], dims[2]))
                end
                Dagger.@spawn transpose(In(a), Out(b))
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[3]), In(dims[3]))
            end
        end
        end

        return collect(b)
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

#in-place    #TODO:Fix pointer error
function fft!(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{FFT!,RFFT!,R2R!}},
    dims::NTuple{N,Int};
    decomp::Union{Pencil,Nothing} = nothing
) where {T,N}
    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)

    if N == 1
        x = size(A, 1)
        a = create_darray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
        end
        end

    return collect(a)

    elseif N == 2
        x, y = size(A)
        if transforms[1] isa RFFT! #R2C
            a = create_darray(A, Blocks(x, div(y, m)))
            buffer = create_darray(A[1:div(x, 2) + 1, :], Blocks(div(x, 2) + 1, div(y, m)))
            b = create_darray(A[1:div(x, 2) + 1, :], Blocks(div(x, (2*n)) + 1, y))
        elseif T <: Real && all(transform -> transform isa FFT!, transforms) #R2C
            a = create_darray(A, Blocks(x, div(y, m)))
            buffer = create_darray(A, Blocks(x, div(y, m)))
            b = create_darray(A, Blocks(div(x, n), y))
        else #C2C or R2R
            a = create_darray(A, Blocks(x, div(y, m)))
            b = create_darray(A, Blocks(div(x, n), y))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT!
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(y))
                end
            elseif T <: Real && all(transform -> transform isa FFT!, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
        end

    end
        return collect(b)

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa RFFT! #R2C
            a =  create_darray(A, Blocks(x, div(y, n), div(z, m)))
            buffer = create_darray((A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2) + 1, div(y, n), div(z, m)))
            b = create_darray((A[1:div(x, 2) + 1, :, :]), Blocks(div(x, (2*n)) + 1, y, div(z, m)))
            c = create_darray((A[1:div(x, 2) + 1, :, :]), Blocks(div(x, (2*n)) + 1, div(y, m), z))
        elseif T <: Real && all(transform -> transform isa FFT!, transforms) #R2C
            a =  create_darray(A, Blocks(x, div(y, n), div(z, m)))
            buffer = create_darray(A, Blocks(x, div(y, n), div(z, m)))
            b = create_darray(A, Blocks(div(x, n), y, div(z, m))) 
            c = create_darray(A, Blocks(div(x, n), div(y, m), z))
        else #C2C
            a =  create_darray(A, Blocks(x, div(y, n), div(z, m)))
            b =  create_darray(A, Blocks(div(x, n), y, div(z, m)))
            c =  create_darray(A, Blocks(div(x, n), div(y, m), z))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT!
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(z))
                end
            elseif T <: Real && all(transform -> transform isa FFT!, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
        
            for idx in 1:length(c.chunks)
                c_part = c.chunks[idx]
                Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[3]), In(dims[3]))
            end
        end
    end
    return collect(c)
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

function fft!(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{FFT!,RFFT!,R2R!}},
    dims::NTuple{N,Int};
    decomp::Decomposition = Slab()
) where {T,N}

    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)

    if N == 1
        x = size(A, 1)
        a = create_darray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
          end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa RFFT #R2C
            a = create_darray(A, Blocks(x, div(y, num_gpus)))
            buffer = create_darray((A[1:div(x, 2) + 1, :]), Blocks(div(x, 2) + 1, div(y, num_gpus)))
            b = create_darray((A[1:div(x, 2) + 1, :]), Blocks(div(x, (2*num_gpus)) + 1, y))
        elseif T <: Real && all(transform -> transform isa FFT, transforms) #R2C
            a = create_darray(A, Blocks(x, div(y, num_gpus)))
            buffer = create_darray(A, Blocks(x, div(y, num_gpus)))
            b = create_darray(A, Blocks(div(x, num_gpus), y))
        else #C2C
            a = create_darray(A, Blocks(x, div(y, num_gpus)))
            b = create_darray(A, Blocks(div(x, num_gpus), y))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(y))
                end
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
          end
        end

        return collect(b)

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa RFFT #R2C
            a =  create_darray(A, Blocks(x, y, div(z, num_gpus)))
            buffer = create_darray((A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2) + 1, y, div(z, num_gpus)))
            b = create_darray((A[1:div(x, 2) + 1, :, :]), Blocks((div(div(x, 2) + 1), 2) +1, div(y, m), z))
        elseif T <: Real && all(transform -> transform isa FFT, transforms) #R2C
            a =  create_darray(A, Blocks(x, y, div(z, num_gpus)))
            buffer = create_darray(A, Blocks(x, y, div(z, W)))
            b = create_darray(A, Blocks(div(x, n), div(y, m), z)) 
        else #C2C
            a =  create_darray(A, Blocks(x, y, div(z, num_gpus)))
            b =  create_darray(A, Blocks(div(x, n), div(y, m), z))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), (dims[1], dims[2]), In(z))
                end
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), (dims[1], dims[2]))
                end
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), (dims[1], dims[2]))
                end
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[3]), In(dims[3]))
            end
          end
        end

        return collect(b)
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

#out-of-place
function ifft(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{IFFT,IRFFT,R2R}},
    dims::NTuple{N,Int};
    decomp::Union{Pencil,Nothing} = nothing
) where {T,N}
    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)

    if N == 1
        x = size(A, 1)
        a = DArray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
            if transforms[1] isa R2R
                a ./= (2 * x)
            end
            end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa IRFFT  # C2R
            a = DArray(A, Blocks(div(x, n), y))
            b = DArray(A, Blocks(x, div(y, m)))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y)), Blocks((x - 1) * 2, div(y, m)))
        else  # C2C or R2R
            a = DArray(A, Blocks(div(x, n), y))
            b = DArray(A, Blocks(x, div(y, m)))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
            end
            if transforms[2] isa R2R
                a ./= (2 * y)
            end

            if transforms[1] isa IRFFT
                Dagger.@spawn transpose(In(a), Out(b))
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), In(transforms[1]), In(dims[1]), In(y))
                end
            else
                Dagger.@spawn transpose(In(a), Out(b))
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[1] isa R2R
                    b ./= (2 * x)
                end
            end
        end
        end

        if transforms[1] isa IRFFT
            return collect(buffer)
        else
            return collect(b)
        end

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa IRFFT  # C2R
            a = DArray(A, Blocks(div(x, n), div(y, m), z))
            b = DArray(A, Blocks(div(x, n), y, div(z, m)))
            c = DArray(A, Blocks(x, div(y, n), div(z, m))) 
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y, z)), Blocks((x - 1) * 2, div(y, n), div(z, m)))
        else  # C2C or R2R
            a = DArray(A, Blocks(div(x, n), div(y, m), z))
            b = DArray(A, Blocks(div(x, n), y, div(z, m)))
            c = DArray(A, Blocks(x, div(y, n), div(z, m))) 
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), In(dims[3]))
            end
            if transforms[1] isa R2R
                a ./= (2 * x)
            end

            Dagger.@spawn transpose(In(a), Out(b))

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
            if transforms[2] isa R2R
                b ./= (2 * y)
            end
            if transforms[1] isa IRFFT
                Dagger.@spawn transpose(In(b), Out(c))
                for idx in 1:length(c.chunks)
                    c_part = c.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(c_part), In(transforms[1]), In(dims[1]), In(z))
                end
            else
                Dagger.@spawn transpose(In(b), Out(c))
                for idx in 1:length(c.chunks)
                    c_part = c.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[3] isa R2R
                    c ./= (2 * z)
                end
            end
        end
        end

        if transforms[1] isa IRFFT
            return collect(buffer)
        else
            return collect(c)
        end
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end


function ifft(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{IFFT,IRFFT,R2R}},
    dims::NTuple{N,Int};
    decomp::Decomposition = Slab()
) where {T,N}
    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)

    if N == 1
        x = size(A, 1)
        a = DArray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
            if transforms[1] isa R2R
                a ./= 2 * x
            end
            end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa IRFFT #R2C
            a = DArray(A, Blocks(x, div(y, num_gpus)))
            b = DArray(A, Blocks(div(x, num_gpus), y))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y)), Blocks(div(((x - 1) * 2), num_gpus), y))
        else #C2C
            a = DArray(A, Blocks(x, div(y, num_gpus)))
            b = DArray(A, Blocks(div(x, num_gpus), y))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
            end
            if transforms[2] isa R2R
                a ./= 2 * y
            end
            if transforms[1] isa IRFFT
                Dagger.@spawn transpose(In(a), Out(b))
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), (IRFFT()), In(dims[1]), In(y))
                end
            else
                Dagger.@spawn transpose(In(a), Out(b))
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[1] isa R2R
                    b ./= 2 * x
                end
            end
        end
        end

        if transforms[1] isa IRFFT
            return collect(buffer)
        else
            return collect(b)
        end

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa IRFFT #R2C
            a =  DArray(A, Blocks(x, y, div(z, num_gpus)))
            b = DArray(A, Blocks(div(x, n), div(y, m), z))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y, z)), Blocks(div(((x - 1) * 2), n), div(y, m), z))
        else #C2C
            a =  DArray(A, Blocks(x, y, div(z, num_gpus)))
            b =  DArray(A, Blocks(div(x, n), div(y, m), z))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), (dims[3], dims[2]))
            end
            if transforms[3] isa R2R
                a ./= (2 * z) * (2 * y)
            end
            if transforms[1] isa IRFFT
                Dagger.@spawn transpose(In(a), Out(b))
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), (IRFFT()), In(dims[1]), In(z))
                end
            else
                Dagger.@spawn transpose(In(a), Out(b))
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[1] isa R2R
                    b ./= 2 * x
                end
            end
            end
        end

        if transforms[1] isa IRFFT
            return collect(buffer)
        else
            return collect(b)
        end
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

#in_place  #TODO: Fix pointer issue
function ifft!(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{IFFT!,IRFFT!,R2R!}},
    dims::NTuple{N,Int};
    decomp::Union{Pencil,Nothing} = nothing
) where {T,N}
    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)

    if N == 1
        x = size(A, 1)
        a = create_darray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
            if transforms[1] isa R2R
                a ./= (2 * x)
            end
            end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa IRFFT  # C2R
            a = create_darray(A, Blocks(div(x, n), y))
            b = create_darray(A, Blocks(x, div(y, m)))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y)), Blocks((x - 1) * 2, div(y, m)))
        else  # C2C or R2R
            a = create_darray(A, Blocks(div(x, n), y))
            b = create_darray(A, Blocks(x, div(y, m)))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
            end
            if transforms[2] isa R2R
                a ./= (2 * y)
            end

            if transforms[1] isa IRFFT
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), In(transforms[1]), In(dims[1]), In(y))
                end
            else
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[1] isa R2R
                    b ./= (2 * x)
                end
            end
            end
        end

        if transforms[1] isa IRFFT
            return collect(buffer)
        else
            return collect(b)
        end

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa IRFFT  # C2R
            a = create_darray(A, Blocks(div(x, n), div(y, m), z))
            b = create_darray(A, Blocks(div(x, n), y, div(z, m)))
            c = create_darray(A, Blocks(x, div(y, n), div(z, m))) 
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y, z)), Blocks((x - 1) * 2, div(y, n), div(z, m)))
        else  # C2C or R2R
            a = create_darray(A, Blocks(div(x, n), div(y, m), z))
            b = create_darray(A, Blocks(div(x, n), y, div(z, m)))
            c = create_darray(A, Blocks(x, div(y, n), div(z, m))) 
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), In(dims[3]))
            end
            if transforms[1] isa R2R
                a ./= (2 * x)
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
            if transforms[2] isa R2R
                b ./= (2 * y)
            end

            if transforms[1] isa IRFFT
                for idx in 1:length(c.chunks)
                    c_part = c.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(c_part), In(transforms[1]), In(dims[1]), In(z))
                end
            else
                for idx in 1:length(c.chunks)
                    c_part = c.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[3] isa R2R
                    c ./= (2 * z)
                end
            end
            end
        end

        if transforms[1] isa IRFFT
            return collect(buffer)
        else
            return collect(c)
        end
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

function ifft!(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{IFFT!,IRFFT!,R2R!}},
    dims::NTuple{N,Int};
    decomp::Decomposition = Slab()
) where {T,N}
    num_gpus = length(CUDA.devices())
    n, m = find_factors(num_gpus)
    scope = Dagger.scope(cuda_gpus=1:num_gpus)

    if N == 1
        x = size(A, 1)
        a = create_darray(A, Blocks(div(x, num_gpus)))

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
            if transforms[1] isa R2R!
                a ./= 2 * x
            end
            end
        end

        return collect(a)

    elseif N == 2
        x, y = size(A)

        if transforms[1] isa IRFFT  # C2R
            a = create_darray(A, Blocks(div(x, n), y))
            b = create_darray(A, Blocks(x, div(y, m)))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y)), Blocks((x - 1) * 2, div(y, m)))
        else  # C2C or R2R
            a = create_darray(A, Blocks(div(x, n), y))
            b = create_darray(A, Blocks(x, div(y, m)))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
            end
            if transforms[2] isa R2R!
                a ./= 2 * y
            end

            if transforms[1] isa IRFFT!
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), (IRFFT!()), In(dims[1]), In(y))
                end
            else
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[1] isa R2R!
                    b ./= 2 * x
                end
            end
            end
        end

        if transforms[1] isa IRFFT!
            return collect(buffer)
        else
            return collect(b)
        end

    elseif N == 3
        x, y, z = size(A)

        if transforms[1] isa IRFFT! #R2C
            a =  create_darray(A, Blocks(x, y, div(z, num_gpus)))
            b = create_darray(A, Blocks(div(x, n), div(y, m), z))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y, z)), Blocks(div(((x - 1) * 2), n), div(y, m), z))
        else #C2C
            a =  create_darray(A, Blocks(x, y, div(z, num_gpus)))
            b =  create_darray(A, Blocks(div(x, n), div(y, m), z))
        end

        Dagger.spawn_datadeps() do
            Dagger.with_options(;scope) do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), (dims[3], dims[2]))
            end
            if transforms[3] isa R2R!
                a ./= (2 * z) * (2 * y)
            end

            if transforms[1] isa IRFFT!
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), (IRFFT!()), In(dims[1]), In(z))
                end
            else
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                if transforms[1] isa R2R!
                    b ./= 2 * x
                end
            end
            end
        end

        if transforms[1] isa IRFFT!
            return collect(buffer)
        else
            return collect(b)
        end
    else
        error("This function only supports 1D, 2D, and 3D arrays")
    end
end

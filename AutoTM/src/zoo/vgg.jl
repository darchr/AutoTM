abstract type AbstractVgg end
struct Vgg19  <: AbstractVgg end
struct Vgg416 <: AbstractVgg end

vgg19() = UnstableChain([
        # First Layer
        Conv((3,3), 3 => 64, relu; pad = 1),
        Conv((3,3), 64 => 64, relu; pad = 1),
        MaxPool((2,2)),
        # Second Layer
        Conv((3,3), 64 => 128, relu; pad = 1),
        Conv((3,3), 128 => 128, relu; pad = 1),
        MaxPool((2,2)),
        # Third Layer
        Conv((3,3), 128 => 256, relu; pad = 1),
        Conv((3,3), 256 => 256, relu; pad = 1),
        Conv((3,3), 256 => 256, relu; pad = 1),
        Conv((3,3), 256 => 256, relu; pad = 1),
        MaxPool((2,2)),
        # Fourth Layer
        Conv((3,3), 256 => 512, relu; pad = 1),
        Conv((3,3), 512 => 512, relu; pad = 1),
        Conv((3,3), 512 => 512, relu; pad = 1),
        Conv((3,3), 512 => 512, relu; pad = 1),
        MaxPool((2,2)),
        # Fifth Layer
        Conv((3,3), 512 => 512, relu; pad = 1),
        Conv((3,3), 512 => 512, relu; pad = 1),
        Conv((3,3), 512 => 512, relu; pad = 1),
        Conv((3,3), 512 => 512, relu; pad = 1),
        MaxPool((2,2)),
        # Fully Connected Layers
        x -> reshape(x, :, size(x, 4)),
        Dense(25088, 4096, relu),
        Dense(4096, 4096, relu),
        Dense(4096, 1000, relu),
        # Truncate to a small positive value to avoid NaNs
        x -> log.(max.(x, Float32(1e-7))),
        softmax
    ])

function vgg416()
    loops = (80, 81, 82, 83, 83)
    #loops = (40, 41, 42, 43, 43)
    layers = []
    # First Layer
    push!(layers, Conv((3,3), 3 => 64, relu; pad = 1))
    for _ in 1:loops[1]
    push!(layers, Conv((3,3), 64 => 64, relu; pad = 1))
    end
    push!(layers,
        Conv((3,3), 64 => 128, relu; pad = 1),
        MaxPool((2,2))
    )

    # Second Layer
    for _ in 1:loops[2]
        push!(layers, Conv((3,3), 128 => 128, relu; pad = 1))
    end
    push!(layers,
        Conv((3,3), 128 => 256, relu; pad = 1),
        MaxPool((2,2))
    )
    # Third Layer
    for _ in 1:loops[3]
        push!(layers, Conv((3,3), 256 => 256, relu; pad = 1))
    end
    push!(layers,
        Conv((3,3), 256 => 512, relu; pad = 1),
        MaxPool((2,2)),
    )

    # Fourth Layer
    for _ in 1:loops[4]
        push!(layers, Conv((3,3), 512 => 512, relu; pad = 1))
    end
    push!(layers, MaxPool((2,2)))
    # Fifth Layer
    for _ in 1:loops[5]
        push!(layers, Conv((3,3), 512 => 512, relu; pad = 1))
    end
    push!(layers,
        MaxPool((2,2)),
        # Fully Connected Layers
        x -> reshape(x, :, size(x, 4)),
        Dense(25088, 4096, relu),
        Dense(4096, 4096, relu),
        Dense(4096, 1000),
        # Add a small positive value to avoid NaNs
        x -> max.(x, Float32(1e-7)),
        softmax
    )

    # Return a vetor version of a chain
    return UnstableChain(layers)
end

function vgg19_inference(batchsize)
    x = rand(Float32, 224, 224, 3, batchsize)

    x = (x .- mean(x)) ./ std(x)

    forward = vgg19()
    return Actualizer(forward, x)
end

_forward(::Vgg19) = vgg19()
_forward(::Vgg416) = vgg416()

function vgg_training(vgg::T, batchsize) where {T <: AbstractVgg}
    X = randn(Float32, 224, 224, 3, batchsize)
    Y = zeros(Float32, 1000, batchsize)
    for col in 1:batchsize
        Y[rand(1:1000), col] = one(eltype(Y))
    end

    # Get the forward pass
    forward = _forward(vgg)

    f = ForwardLoss(_forward(vgg), Flux.crossentropy)
    return Actualizer(f, X, Y; optimizer = nGraph.SGD(Float32(0.05)))
end

function random_labels!(y::AbstractArray{T,2}) where {T}
    y .= zero(eltype(y))
    for j in 1:size(y, 2)
        y[rand(1:size(y, 1)), j] = one(eltype(y))
    end
end

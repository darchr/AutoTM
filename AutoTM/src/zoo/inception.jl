# We're going to create a struct that executes a bunch of parallel branches and then
# concatenates the results back together
struct InceptionBlock{T <: Tuple}
    branches::T
end

InceptionBlock(x, y, args...) = InceptionBlock((x, y, args...))

# Flux likes the "@treelike" macro for doing backpropogation.
Flux.@functor InceptionBlock

function (block::InceptionBlock)(x)
    # Call branches
    results = map(f -> f(x), block.branches)
    return cat(results...; dims = 3)
end

const A_SIZE = (35, 35, 384)
const B_SIZE = (17, 17, 1024)
const C_SIZE = (8, 8, 1536)

# We're gonna be REALLY mean to type inference. Sorry Julia!
function stem()
    #println("Stem")

    return Chain(
        # Initial Stem
        Conv((3, 3), 3 => 32, relu; pad = 0, stride = 2),
        Conv((3, 3), 32 => 32, relu; pad = 0),
        Conv((3, 3), 32 => 64, relu; pad = 1),

        # First Split
        InceptionBlock(
            MaxPool((3,3); pad = 0, stride = 2),
            Conv((3,3), 64 => 96, relu; pad = 0, stride = 2),
        ),

        # Second Split
        InceptionBlock(
            Chain(
                Conv((1,1), 160 => 64, relu; pad = 0, stride = 1),
                Conv((3,3), 64 => 96, relu; pad = 0),
            ),
            Chain(
                Conv((1,1), 160 => 64, relu; pad = 0),
                Conv((7,1), 64 => 64, relu; pad = (3, 3, 0, 0)),
                Conv((1,7), 64 => 64, relu; pad = (0, 0, 3, 3)),
                Conv((3,3), 64 => 96, relu; pad = 0),
            ),
        ),

        # Final Split
        InceptionBlock(
            Conv((3,3), 192 => 192, relu; pad = 0, stride = 2),
            MaxPool((3,3); pad = 0, stride = 2),
        )
    )
end

function inception_a()
    #println("A Block")
    return InceptionBlock(
        Chain(
            x -> meanpool(x, (3,3); pad = 1, stride = 1),
            Conv((1,1), 384 => 96, relu; pad = 0)
        ),
        Conv((1,1), 384 => 96, relu; pad = 0),
        Chain(
            Conv((1,1), 384 => 64, relu; pad = 0),
            Conv((3,3), 64 => 96, relu; pad = 1)
        ),
        Chain(
            Conv((1,1), 384 => 64, relu; pad = 0),
            Conv((3,3), 64 => 96, relu; pad = 1),
            Conv((3,3), 96 => 96, relu; pad = 1)
        )
    )
end

function inception_b()
    #println("B Block")
    S = B_SIZE[3]
    return InceptionBlock(
        Chain(
            x -> meanpool(x, (3,3); pad = 1, stride = 1),
            Conv((1,1), S => 128)
        ),
        Conv((1,1), S => 384, relu),
        Chain(
            Conv((1,1), S => 192, relu),
            Conv((1,7), 192 => 224, relu; pad = (0, 0, 3, 3)),
            Conv((7,1), 224 => 256, relu; pad = (3, 3, 0, 0)),
        ),
        Chain(
            Conv((1,1), S => 192, relu),
            Conv((1,7), 192 => 192, relu; pad = (0, 0, 3, 3)),
            Conv((7,1), 192 => 224, relu; pad = (3, 3, 0, 0)),
            Conv((1,7), 224 => 224, relu; pad = (0, 0, 3, 3)),
            Conv((7,1), 224 => 256, relu; pad = (3, 3, 0, 0)),
        )
    )
end

function inception_c()
    #println("C Block")
    S = C_SIZE[3]
    return InceptionBlock(
        Chain(
            x -> meanpool(x, (3,3); pad = 1, stride = 1),
            Conv((1,1), S => 256, relu)
        ),
        Conv((1,1), S => 256, relu),
        Chain(
            Conv((1, 1), S => 384, relu),
            InceptionBlock(
                Conv((1, 3), 384 => 256, relu; pad = (0, 0, 1, 1)),
                Conv((3, 1), 384 => 256, relu; pad = (1, 1, 0, 0)),
            ),
        ),
        Chain(
            Conv((1,1), S => 384, relu),
            Conv((1,3), 384 => 448, relu; pad = (0, 0, 1, 1)),
            Conv((3,1), 448 => 512, relu; pad = (1, 1, 0, 0)),
            InceptionBlock(
                Conv((1,3), 512 => 256, relu; pad = (0, 0, 1, 1)),
                Conv((3,1), 512 => 256, relu; pad = (1, 1, 0, 0)),
            ),
        )
    )
end

function inception_ra(k, l, m, n)
    #println("A Reduction")
    S = A_SIZE[3]

    return InceptionBlock(
        x -> maxpool(x, (3,3); pad = 0, stride = 2),
        Conv((3,3), S => n, relu; pad = 0, stride = 2),
        Chain(
            Conv((1,1), S => k, relu),
            Conv((3,3), k => l, relu; pad = 1),
            Conv((3,3), l => m, relu; pad = 0, stride = 2)
        ),
    )
end

function inception_rb()
    #println("B Reduction")
    S = B_SIZE[3]

    return InceptionBlock(
        x -> maxpool(x, (3,3); pad = 0, stride = 2),

        Chain(
            Conv((1,1), S => 192, relu),
            Conv((3,3), 192 => 192, relu; pad = 0, stride = 2)
        ),

        Chain(
            Conv((1,1), S => 256, relu; pad = 0),
            Conv((1,7), 256 => 256, relu; pad = (0, 0, 3, 3)),
            Conv((7,1), 256 => 320, relu; pad = (3, 3, 0, 0)),
            Conv((3,3), 320 => 320, relu; pad = 0, stride = 2)
        ),
    )
end

# Legacy from when I didn't know the sizes of the intermediate layers.
#
# Removed the automatic size tracking because it took FOREVER to run the functions
# in the basic NNlib implementations.
mutable struct SizeTracker
    layers::Vector{Any}
    array::Any
end

function Base.push!(S::SizeTracker, f, call = true)
    # This is kind of mindbending ...
    #
    # We call the provided function to get another function that we append onto the
    # layers. Then, we call that generated function to get the new array size.
    if call
        push!(S.layers, f())
    else
        push!(S.layers, f)
    end
end

function inception_v4(x)
    layers = SizeTracker([], x)
    push!(layers, stem)
    for _ in 1:4
        push!(layers, inception_a)
    end
    push!(layers, () -> inception_ra(192, 224, 256, 384))

    for _ in 1:7
        push!(layers, inception_b)
    end
    push!(layers, inception_rb)

    for _ in 1:3
        push!(layers, inception_c)
    end

    #kernel_size = size.(Ref(layers.array), (1, 2))
    kernel_size = (C_SIZE[1], C_SIZE[2])
    push!(layers, x -> meanpool(x, kernel_size; pad = 0, stride = 1), false)
    # dropout

    push!(layers, x -> reshape(x, :, size(x,4)), false)
    push!(layers, Dense(1536, 1000, relu), false)
    push!(layers, x -> log.(max.(x, Float32(1e-7))), false),
    push!(layers, softmax, false)

    return UnstableChain(layers.layers)
end

function inception_v4_training(batchsize; kw...)
    x = rand(Float32, 299, 299, 3, batchsize)
    X = (x .- mean(x)) ./ std(x)

    Y = zeros(Float32, 1000, batchsize)
    random_labels!(Y)

    f = ForwardLoss(inception_v4(x), Flux.crossentropy)
    return Actualizer(f, X, Y; optimizer = nGraph.SGD(Float32(0.05)))
end


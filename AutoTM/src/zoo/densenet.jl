dense_conv(n, k, k0) = Chain(
    BatchNorm(n * k + k0, relu),
    # The 1x1 convolutions produce 4K feature maps
    Conv( (1,1), n * k + k0 => 4 * k),
    BatchNorm(4 * k, relu),
    Conv( (3,3), 4 * k => k; pad = 1)
)

struct DenseBlock
    layers::Vector
    transition
    apply_transition::Bool
end

Flux.@treelike DenseBlock

function (D::DenseBlock)(x)
    for f in D.layers
        y = f(x)
        x = cat(x, y; dims = 3)
    end
    # Transition Layer
    ret = D.apply_transition ? D.transition(x) : x
    return ret
end

function DenseBlock(k::Int, k0, N::Int; apply_transition = true)
    layers = [dense_conv(i-1, k, k0) for i in 1:N]

    transition = Chain(
        BatchNorm( N * k + k0, relu),
        Conv( (1,1), N * k + k0 => k0),
        x -> meanpool(x, (2,2); pad = 0, stride = 2),
    )
    return DenseBlock(layers, transition, apply_transition)
end

function DenseNet(k) 
    counts = (6, 12, 64, 48)
    return Chain(
        Conv( (7,7), 3 => 2 * k, relu; pad = 3, stride = 2),
        x -> maxpool(x, (3,3); pad = 1, stride = 2),
        DenseBlock(k, 2*k, counts[1]),
        DenseBlock(k, 2*k, counts[2]),
        DenseBlock(k, 2*k, counts[3]),
        DenseBlock(k, 2*k, counts[4]; apply_transition = false),
        x -> meanpool(x, (7,7); pad = 0),
        x -> reshape(x, :, size(x,4)),
        Dense(counts[4] * k + 2 * k, 1000),
        x -> log.(max.(x, Float32(1e-9))),
        softmax
    )
end

function densenet_training(batchsize; growth = 32)
    x = rand(Float32, 224, 224, 3, batchsize)
    x = (x .- mean(x)) ./ std(x)

    y = rand(Float32, 1000, batchsize)
    random_labels!(y) 
    forward = DenseNet(growth)
    f(x, y) = Flux.crossentropy(forward(x), y)
    return Actualizer(f, x, y; optimizer = nGraph.SGD(Float32(0.001)))
end

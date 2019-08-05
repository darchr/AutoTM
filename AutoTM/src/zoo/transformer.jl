# Scaled dot-product attention function
function scaled_dot_product_attention(q, k, v) 
    x = nGraph.bmm(q, k; transpose_b = true)
    x = x ./ size(k, 1)
    x = Flux.softmax(x)
    return nGraph.bmm(x, v)
end

# multi-head attention
#
# Arguments `q`, `k`, and `v` are the queries, keys, and values
# Arguments `wq`, `wk`, and `wv` vectors of projection matrices
function multi_head_attention(q, k, v, wq, wk, wv, wo)
    heads = map(zip(wq, wk, wv)) do x
        # Do a broadcast operation 
        return scaled_dot_product_attention(x[1] * q, x[2] * k, x[3] * v)
    end

    return wo * cat(heads...; dims = 1)
end

# Point wise feed-forward networks
feed_forward(x, W::Vector, B::Vector) = W[2] * Flux.relu(W[1] * x .+ B[1]) .+ B[2]

function layer_normalization(a, g, b)
    μ = sum(a) ./ size(a, 1)
    diff = a .- μ
    σ = sqrt(sum((diff .* diff) ./ size(a, 1)))
    return ((g ./ σ) .* diff) .+ b
end

#####
##### Helpers for building the whole network
#####
struct AttentionLayer
    wq::Vector
    wk::Vector
    wv::Vector
    wo
end

function AttentionLayer(dmodel, h)
    dk = div(dmodel, h)
    wq = [param(randn(Float32, dk, dmodel)) for _ in 1:h]
    wk = [param(randn(Float32, dk, dmodel)) for _ in 1:h]
    wv = [param(randn(Float32, dk, dmodel)) for _ in 1:h]
    wo = param(randn(Float32, dmodel, dmodel))

    return AttentionLayer(wq, wk, wv, wo)
end

(A::AttentionLayer)(q, k, v) = multi_head_attention(q, k, v, A.wq, A.wk, A.wv, A.wo)

struct FeedForward
    W::Vector
    B::Vector
end

function FeedForward(dmodel::Integer, dff::Integer)
    W1 = param(randn(Float32, dff, dmodel))
    B1 = param(randn(Float32, dff))

    W2 = param(randn(Float32, dmodel, dff))
    B2 = param(randn(Float32, dmodel))
    return FeedForward([W1, W2], [B1, B2])
end

(F::FeedForward)(x) = feed_forward(x, F.W, F.B)

struct Normalize
    g
    b
end

function Normalize(dmodel::Integer)
    g = param(randn(Float32, dmodel))
    b = param(randn(Float32, dmodel))
    return Normalize(g, b)
end

(N::Normalize)(x) = layer_normalization(x, N.g, N.b)

# Encoder Layers
struct Encoder
    attentions::Vector{AttentionLayer}
    att_normalize::Vector{Normalize}
    forwards::Vector{FeedForward}
    fwd_normalize::Vector{Normalize}
end

function Encoder(N::Integer, dmodel::Integer, dff::Integer, h::Integer)
    attentions = [AttentionLayer(dmodel, h) for _ in 1:N]
    att_normalize = [Normalize(dmodel) for _ in 1:N]
    forwards = [FeedForward(dmodel, dff) for _ in 1:N]
    fwd_normalize = [Normalize(dmodel) for _ in 1:N]
    return Encoder(attentions, att_normalize, forwards, fwd_normalize)
end

function (E::Encoder)(x)
    iter = zip(E.attentions, E.att_normalize, E.forwards, E.fwd_normalize)
    for (attention, att_n, forward, fwd_n) in iter
        x = att_n(x + attention(x, x, x))
        x = fwd_n(x + forward(x))
    end
    return x
end

# Decoder Layers
struct Decoder
    first_attentions::Vector{AttentionLayer}
    first_normalize::Vector{Normalize}
    second_attentions::Vector{AttentionLayer}
    second_normalize::Vector{Normalize}
    forwards::Vector{FeedForward}
    fwd_normalize::Vector{Normalize}
end

function Decoder(N::Integer, dmodel::Integer, dff::Integer, h::Integer)
    first_attentions = [AttentionLayer(dmodel, h) for _ in 1:N]
    first_normalize = [Normalize(dmodel) for _ in 1:N]
    second_attentions = [AttentionLayer(dmodel, h) for _ in 1:N]
    second_normalize = [Normalize(dmodel) for _ in 1:N]
    forwards = [FeedForward(dmodel, dff) for _ in 1:N]
    fwd_normalize = [Normalize(dmodel) for _ in 1:N]

    return Decoder(
        first_attentions, 
        first_normalize,
        second_attentions, 
        second_normalize,
        forwards,
        fwd_normalize
    )
end

function (D::Decoder)(x, hidden)
    iter = zip(
        D.first_attentions, 
        D.first_normalize,
        D.second_attentions, 
        D.second_normalize,
        D.forwards,
        D.fwd_normalize
    )
    for (first_attention, first_n, second_attention, second_n, forward, fwd_n) in iter
        x = first_n(x + first_attention(x, x, x))
        x = second_n(x + second_attention(hidden, hidden, x))
        x = fwd_n(x + forward(x))
    end
    return x
end

# Top level entry point
struct Transformer
    encoder::Encoder
    decoder::Decoder
    dense::Flux.Dense
end

function Transformer(N::Integer, dmodel::Integer, dff::Integer, h::Integer, vocab, seqlen)
    encoder = Encoder(N, dmodel, dff, h)
    decoder = Decoder(N, dmodel, dff, h)
    dense = Flux.Dense(dmodel * seqlen, vocab)
    return Transformer(encoder, decoder, dense)
end

function (T::Transformer)(inputs, outputs, target)
    hidden = T.encoder(inputs)
    y = T.decoder(outputs, hidden)
    # Reshape the first two dimensions into one dimension
    y = reshape(y, :, size(y, 3))
    y = Flux.softmax(T.dense(y))

    # Generate a loss
    return Flux.crossentropy(y, target)
end

function transformer_training(batchsize = 16, seq_length = 50; backend = nGraph.Backend())
    # Parameters from the Transformer Paper

    # Number of Stacked Layers
    N = 8
    # Size of the first tensor dimension
    dmodel = 512
    # Hidden size for the feed forward networks
    dff = 2048 
    # Number of shards for the attention layers
    h = 8
    # Output vocab size
    vocab = 25000 

    # Create inputs, outputs, and expected output layers
    X = nGraph.Node(randn(Float32, dmodel, seq_length, batchsize))
    Y = nGraph.Node(randn(Float32, dmodel, seq_length, batchsize))
    expected = randn(Float32, vocab, batchsize)
    random_labels!(expected)
    expected = nGraph.Node(expected)

    T = Transformer(N, dmodel, dff, h, vocab, seq_length)
    
    kw = (optimizer = nGraph.SGD(Float32(0.001)),)
    return T, (X, Y, expected), kw
end


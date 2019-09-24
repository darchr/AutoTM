# Construct the various pieces of the DLRM as functions.
struct DLRM
    bottom_mlp
    embedding_tables
    interaction
    top_mlp
end

function create_dlrm(;
        m_sparse,       # Output feature size of the embedding tables.
        ln_emb,         # collection of "n" sizes for the embedding tables.
        ln_bot,         # dimensions for the bottom mlp
        ln_top,         # dimensions for the top mlp
        arch_interaction_op,        # interaction between sparse and dense dimensions
        sigmoid_bot = -1,
        sigmoid_top = -1,
        loss_threshold = 0.0,
    )

    # Create embedding layers
    embedding_tables = create_embedding(m_sparse, ln_emb)
    bottom_mlp = create_mlp(ln_bot, sigmoid_bot)
    interaction = arch_interaction_op
    top_mlp = create_mlp(ln_top, sigmoid_top)

    return DLRM(
        bottom_mlp,
        embedding_tables,
        interaction,
        top_mlp
    )
end

function runlayers(layers, x)
    for l in layers
        x = l(x)
    end
    return x
end

function (D::DLRM)(dense_input, sparse_inputs...)
    # Run the bottom MLP
    dense_features = runlayers(D.bottom_mlp, dense_input)

    # Run the embedding tables.
    sparse_features = []
    for (i, indices) in enumerate(sparse_inputs)
        push!(sparse_features, D.embedding_tables[i](indices))
    end

    # Run the interaction.
    Z = D.interaction(dense_features, sparse_features) 

    # Run the top MLP
    return runlayers(D.top_mlp, Z)
end

function create_embedding(sparse_dim::Integer, sizes)
    tables = []
    for sz in sizes
        # Create the table.
        table = Flux.param(randn(Float32, sparse_dim, sz))

        # The Facebook implementations use the embedding bag approach, but looking at the 
        # output is is just performing a standard embedding - so that's what we do here.
        f = x -> nGraph.embedding(x, nGraph.Node(table))
        push!(tables, f)
    end
    return tables
end

function create_mlp(sizes, sigmoid_layer)
    layers = []  
    for i in 1:length(sizes) - 1
        m = sizes[i]
        n = sizes[i+1]

        if i == sigmoid_layer
            σ = Flux.sigmoid
        else
            σ = Flux.relu
        end
        L = Dense(m, n, σ)
        push!(layers, L)
    end
    return layers
end

function kaggle_dlrm(;m_sparse = 16, batchsize = 128, dense_input_size = 13)
    # Extracted from the Facebook implementation.
    embedding_sizes = 5 .* [
        1461     ,
        586      ,   
        10131227 ,   
        2202608  ,   
        306      ,   
        24       ,   
        12518    ,   
        634      ,   
        4        ,   
        93146    ,   
        5684     ,   
        8351593  ,   
        3195     ,   
        28       ,   
        14993    ,   
        5461306  ,   
        11       ,   
        5653     ,   
        2173     ,   
        4        ,   
        7046547  ,   
        18       ,   
        16       ,   
        286181   ,   
        105      ,   
        142572   ,   
    ]

    bottom_mlp_sizes = [13, 512, 256, 64, 16]
    bottom_mlp_sigmoid = -1

    # Replace the first entry with 745 becuase we aren't able to do the triangular selection
    # thing easily in nGraph.
    top_mlp_sizes = [745, 512, 256, 1]
    top_mlp_sigmoid = 3

    interaction = dot_interaction

    dlrm = create_dlrm(;
        m_sparse = m_sparse,
        ln_emb = embedding_sizes,
        ln_bot = bottom_mlp_sizes,
        ln_top = top_mlp_sizes,
        arch_interaction_op = interaction,
        sigmoid_bot = bottom_mlp_sigmoid,
        sigmoid_top = top_mlp_sigmoid,
    )

    backend = nGraph.Backend("CPU")

    # Create appropriate inputs.
    dense_input = randn(Float32, dense_input_size, batchsize)
    sparse_inputs = [rand(Int32(1):Int32(e), batchsize) for e in embedding_sizes]

    loss(y, x...) = Flux.crossentropy(dlrm(x...), y)

    expected_values = rand((Float32(0),Float32(1)), 1, batchsize)
    kw = (optimizer = nGraph.SGD(Float32(0.1)),)

    return loss, (expected_values, dense_input, sparse_inputs...), kw
end

#####
##### dot interactions
#####

_extrude_reshape(x) = reshape(x, size(x, 1), 1, size(x, 2))

function dot_interaction(dense_features, sparse_features)
    # Concatenate features along the second dimension
    # Reshape everything to a 3d vector
    T′ = dense_features
    for S in sparse_features
        T′ = cat(T′, S; dims = 1)
    end
    T = reshape(
        T′,
        size(dense_features, 1),
        :, 
        size(dense_features, 2),
    )

    @show size(T)
    # Perform a dot product
    Z = nGraph.bmm(T, T; transpose_b = true)
    Z = reshape(Z, :, size(Z, 3))

    @show size(dense_features)
    @show size(Z)

    # Change everything to a row vector - concatenate dense features onto the 
    return cat(dense_features, Z; dims = 1)
end

module Experiments

import ..Zoo: Zoo

# Make a data directory if needed
function __init__()
    for dir in (DATADIR, CACHEDIR)
        !ispath(dir) && mkdir(dir)
    end
end

# Common Paths
const EXPERIMENTS_DIR = @__DIR__
const SRCDIR = dirname(EXPERIMENTS_DIR)
const PKGDIR = dirname(SRCDIR)
const REPODIR = dirname(PKGDIR)
const DATADIR = joinpath(REPODIR, "data")
const CACHEDIR = joinpath(DATADIR, "caches")

const CPU_CACHE = joinpath(CACHEDIR, "single_cpu_profile.jls")
const GPU_CACHE = joinpath(CACHEDIR, "gpu_profile.jls")

#####
##### Convenience wrappers for common models
#####

conventional_inception()    = Inception_v4(1024)
conventional_resnet()       = Resnet200(512)
conventional_vgg()          = Vgg19(2048)
conventional_densenet()     = DenseNet(512)
conventional_transformer()  = Transformer(512, 150)
test_vgg() = Vgg19(32)

# Large Models
large_inception() = Inception_v4(6144)      # 659 GB
large_vgg() = Vgg416(320)                   # 658 GB
large_resnet() = Resnet200(2560)            # 651 GB
large_densenet() = DenseNet(3072)           # 688 GB

#####
##### Model Implementations
#####

# Resnet
struct Resnet{T}
    batchsize::Int
    zoo::T
end
Resnet50(batchsize) = Resnet(batchsize, Zoo.Resnet50())
Resnet200(batchsize) = Resnet(batchsize, Zoo.Resnet200())

_sz(::Zoo.Resnet50) = "50"
_sz(::Zoo.Resnet200) = "200"
name(R::Resnet) = "resnet$(_sz(R.zoo))_batchsize_$(R.batchsize)"
titlename(R::Resnet) = "Resnet$(_sz(R.zoo))"
(R::Resnet)() = Zoo.resnet_training(R.zoo, R.batchsize)

# VGG
struct Vgg{T}
    batchsize::Int
    zoo::T
end
Vgg19(batchsize) = Vgg(batchsize, Zoo.Vgg19())
Vgg416(batchsize) = Vgg(batchsize, Zoo.Vgg416())

_sz(::Zoo.Vgg19) = "19"
_sz(::Zoo.Vgg416) = "416"
name(R::Vgg) = "vgg$(_sz(R.zoo))_batchsize_$(R.batchsize)"
titlename(R::Vgg) = "Vgg$(_sz(R.zoo))"
(R::Vgg)() = Zoo.vgg_training(R.zoo, R.batchsize)

# Inception
struct Inception_v4
    batchsize::Int
end
name(R::Inception_v4) = "inception_v4_batchsize_$(R.batchsize)"
titlename(::Inception_v4) = "Inception v4"
(R::Inception_v4)() = Zoo.inception_v4_training(R.batchsize)

# DenseNet
struct DenseNet
    batchsize::Int
end
titlename(R::DenseNet) = "DenseNet 264"
name(R::DenseNet) = "densenet264_batchsize_$(R.batchsize)"
(R::DenseNet)() = Zoo.densenet_training(R.batchsize)

struct Transformer
    batchsize::Int
    sequence_length::Int
end
titlename(T::Transformer) = "Transformer"
name(T::Transformer) = "transformer_batchsize_$(T.batchsize)_seqlen_$(T.sequence_length)"
(T::Transformer)() = Zoo.transformer_training(T.batchsize, T.sequence_length)

end #module


# Uncompress the test cache
cachedir = joinpath(@__DIR__, "..", "test", "cache")
tarball = "fuzz.tar.gz"
serialfile = "fuzz.jls"

run(`tar xvf $(joinpath(cachedir, tarball))`)
mv(joinpath(@__DIR__, serialfile), joinpath(cachedir, serialfile); force = true)

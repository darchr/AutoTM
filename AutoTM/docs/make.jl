using Documenter, AutoTM

makedocs(
    modules = [AutoTM],
    format = :html,
    checkdocs = :exports,
    sitename = "AutoTM.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/hildebrandmw/AutoTM.jl.git",
)

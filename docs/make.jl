using Documenter

makedocs(
    modules = [Documenter],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    sitename = "AutoTM",
    doctest = false,
    pages = Any[
        "AutoTM" => "index.md",
        "Installation" => "installation.md",
        "AutoTM Artifact Workflow" => "benchmarker.md",
    ],
)

deploydocs(
    repo = "github.com/darchr/AutoTM.git",
    target = "build",
)

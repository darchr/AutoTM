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
        "Software Requirements" => "software.md",
        "Installation" => "installation.md",
        "AutoTM Artifact Workflow" => "benchmarker.md",
        "Experiment Customization" => "customization.md"
    ],
)

deploydocs(
    repo = "github.com/darchr/AutoTM.git",
    target = "build",
)

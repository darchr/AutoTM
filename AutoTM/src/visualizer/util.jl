#####
##### Plotting Utility Functions
#####

rectangle(x, y, w, h) = (x .+ [0, w, w, 0]), (y .+ [0, 0, h, h])
getname(v, s::Symbol) = getindex.(v, s)

load_save_files(f, formulations::String) = first(load_save_files(f, (formulations,)))
function load_save_files(f, formulations)
    savefiles = [joinpath(savedir(f), join((name(f), i), "_") * ".jls") for i in formulations]
    data = deserialize.(savefiles)
    for d in data
        sort!(d.runs; rev = true, by = x -> get(x, :dram_limit, 0))
    end

    return data
end

# For drawing a vertical asymptote on a graph
vasymptote() = """
\\pgfplotsset{vasymptote/.style={
    before end axis/.append code={
        \\draw[densely dashed] ({rel axis cs:0,0} -| {axis cs:#1,0})
        -- ({rel axis cs:0,1} -| {axis cs:#1,0});
}
}}
"""

hasymptote() = """
\\pgfplotsset{hasymptote/.style={
    before end axis/.append code={
        \\draw[densely dashed] ({rel axis cs:0,0} -| {axis cs:0,#1})
        -- ({rel axis cs:1,0} -| {axis cs:0,#1});
}
}}
"""

hline(y; xl = 0, xu = 1, color = "red") = """
\\draw[$color, densely dashed, ultra thick] L
    ({axis cs:$xl,$y} -| {rel axis cs:0,0}) -- 
    ({axis cs:$xu,$y} -| {rel axis cs:1,0});
""" |> rm_newlines

vline(x; yl = 0, yu = 1, color = "red") = """
\\draw[$color, sharp plot] ($x, $yl) -- ($x, $yu);
""" |> rm_newlines

comment(x, y, str; kw...) = @pgf(["\\node[align = center$(format(kw))] at ", Coordinate(x, y), "{$str};"])
format(kw) = ", " * join(["$a = $b" for (a,b) in kw], ", ")

rm_newlines(str) = join(split(str, "\n"))

# Node - must sort data before hand
# using load_save_files does this automatically
get_dram_performance(data) = minimum(get_dram_performance.(data))
get_dram_performance(data::NamedTuple) = first(getname(data.runs, :actual_runtime))

get_pmm_performance(data) = maximum(get_pmm_performance.(data))
get_pmm_performance(data::NamedTuple) = last(getname(data.runs, :actual_runtime))

function findabsmin(f, x)
    _, ind = findmin(abs.(f.(x)))
    return ind
end


# Tool for converting the old nested dataformat to a DataFrame.
using DataFrames, Serialization, AutoTM, Dates

# Hooks for turning specific structures into a generalized form that don't rely on AutoTM
# specific encoding
#
# General objects will just fallback to themselves
autotm_convert(x) = x

# Define conversion to recurse into collections
autotm_convert(x::Dict) = Dict(autotm_convert(k) => autotm_convert(v) for (k,v) in x)
autotm_convert(x::Vector) = autotm_convert.(x)

# Turn IOConfig into NamedTuples
enum_convert(x) = (x == AutoTM.Utils.DRAM) ? :DRAM : :PMEM

autotm_convert(@nospecialize x::AutoTM.Utils.IOConfig) = (
    inputs = map(enum_convert, x.inputs),
    outputs = map(enum_convert, x.outputs)
)

# Strip off "Ref" types
autotm_convert(x::Ref) = x[]

### The field of the original data type where the run data may be found.
datafield() = :runs

function convert_dir(olddir, newdir)
    files = readdir(olddir)
    for (i, file) in enumerate(files)
        println("On file $i of $(length(files))")
        data_convert(olddir, file, newdir)
    end
end

function data_convert(olddir, filename, newdir)
    # Deserialize the data  
    oldfile = joinpath(olddir, filename)
    X = deserialize(oldfile)
    df = _convert(X)
    newfile = joinpath(newdir, filename)
    serialize(newfile, df)
end

# Inner method - use function barrier for slightly better performance - or something
function _convert(X)
    # Get the global field names.
    globals = filter(!isequal(datafield()), collect(fieldnames(typeof(X))))
    global_data = Dict(f => getfield(X, f) for f in globals)

    # Now, extract the runs
    data = getfield(X, datafield()) 
    nentries = length(data)

    # Merge original run data with the global fields.
    merge!.(data, Ref(global_data))

    # Convert leaf structures into generic types defined in Base for compatibility.
    data = autotm_convert(data)

    # Extract all of the keys from the data
    ks = reduce(union, keys.(data))

    # Make a super dictionary.
    data = Dict(k => get.(data, Ref(k), Ref(missing)) for k in ks)

    # Retroactively add a date field in needed
    if !in(:date, ks)
        data[:date] = [now() for _ in 1:nentries]
    end

    # Finally, create a dataframe from this new data and return it.
    return DataFrame(data)
end

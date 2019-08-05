using Pkg; Pkg.activate("..")
using ArgParse, Benchmarks, Serialization

function main(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nthreads"
            arg_type = Int
            required = true
            help = "Select the number of threads"
        "--datafile"
            arg_type = String
            required = true
            help = "File where the serialized data is held"
        "--refresh"
            action = :store_true
            help = "Start a new data collection"
    end

    parsed_args = parse_args(args, s)

    # Run the benchmark
    data = Benchmarks.benchmark(Kernel, threads = parsed_args["nthreads"]) 

    # If this is not a refresh, deserialize the datafile
    # Apparently "datafile" is stored as an array ...
    datafile = parsed_args["datafile"]
    if !parsed_args["refresh"]
        past_data = deserialize(datafile)
        push!(past_data, data)
        data_to_save = past_data
    else
        data_to_save = [data]
    end

    serialize(datafile, data_to_save)
    return 0
end

main(ARGS)

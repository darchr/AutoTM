# Sort NamedTuples alphabetically
ntsort(x::NamedTuple) = (;sort([pairs(x)...]; by = x -> first(x))...)

_keys(::Type{<:NamedTuple{names}}) where {names} = names

database() = StructArray()

function ismatch(a::NamedTuple, b::NamedTuple)
    # Can only match if `b` is possibly a subset of `a`.
    issubset(keys(b), keys(a)) || return false
    for (k, v) in pairs(b)
        # The "data" field is special - don't compare here for equality
        k == :data && continue
        a[k] == v || return false
    end
    return true
end

# Add in `nothing` entries
function expand(@nospecialize(nt::NamedTuple), names; f = () -> nothing)
    newnames = Tuple(setdiff(names, keys(nt)))
    missing_tuple = NamedTuple{newnames}(ntuple(i -> f(), length(newnames)))
    expanded_nt = merge(nt, missing_tuple)
    return ntsort(expanded_nt)
end

# Hook for customizing merging.
datamerge(a, b) = b

function addentry!(
        db::StructArray,
        @nospecialize(nt::NamedTuple),
        data::Dict{Symbol} = Dict{Symbol,Any}())

    data = Dict{Symbol,Any}(data)

    # This field is reserved. Keep myself from being an idiot.
    @assert !haskey(nt, :data)

    # Expand the new entry so it has all the keys in the original database.
    nt = expand(merge(nt, (data = data,)), _keys(eltype(db)))

    # If the names of `nt` are a subset of the keys already in `db` - we may be able to merge.
    if issubset(keys(nt), _keys(eltype(db)))
        for row in Tables.rows(db)
            if ismatch(row, nt)
                # Merge the data entries
                merge!(datamerge, row.data, nt.data)
                return db
            end
        end
    end

    # If the database starts as empty, just wrap up the current entry
    iszero(length(db)) && return StructArray([nt])

    # Otherwise, we have to do a bunch of promotion on everything.
    arrays = expand(
        StructArrays.fieldarrays(db),
        keys(nt);
        f = () -> fill(nothing, length(db))
    )

    # Make sure the directions here are still equal.
    @assert keys(arrays) == keys(nt)
    conversions = cpush!.(Tuple(arrays), Tuple(nt))

    arrays = NamedTuple{keys(arrays)}(conversions)
    return StructArray(arrays)
end

# Type expanding version push!
cpush!(a::Vector{T}, b::V) where {T, V} = push!(convert(Vector{promote_type(T,V)}, a), b)

function showdb(db)
    sch = Tables.schema(db)
    header = filter(!isequal(:data), collect(sch.names))

    vectors = (getproperty(db, i) for i in header)
    pretty_table(stdout, hcat(vectors...), header)
end


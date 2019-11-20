function ntsorter(x::NamedTuple{names}) where {names}
    sorted_names = Tuple(sort(collect(names)))
    sorted_values = ntuple(i -> x[sorted_names[i]], length(names))
    return NamedTuple{sorted_names}(sorted_values)
end

# Here, we collect all of the statistics we want about each trial.
#
# The parameters fields should be a NamedTuple of run parameters parameters.
#
# We'll keep an invariant that we'll maintain all params across all runs.
# If more parameters are added later, we can must iterate over all entries and update their
# NamedTuple fields with `missing`s.
#
# The `data` field contains collected data and should be expanded as more data is collected
# for the same set of parameters.
mutable struct TableEntry{names}
    created::DateTime
    updated::DateTime
    params::NamedTuple{names}
    data::Dict{Symbol,Any}
end

# Initialize with an empty row - names must be sorted
function TableEntry(params::NamedTuple{names}, data = Dict{Symbol,Any}) where {names}
    @assert issorted(names)
    TableEntry{names}(now(), now(), params, data)
end

# For filtering
function ismatch(T::TableEntry{N}, params::NamedTuple{M}) where {N,M}
    # the names for the parameters must be a subset of the names in the table entry
    issubset(M,N) || return false
    for (k, v) in pairs(params)
        if getproperty(T, k) !== v
            return false
        end
    end
    return true
end
ismatch(T::TableEntry, U::TableEntry) where {N} = ismatch(T, U.params)

# Tables.jl interface for TableEntry (Rows in the Table)
function Base.getproperty(T::TableEntry{names}, name::Symbol) where {names}
    # If the provided name belongs to the NamedTuple of parameters, dispatch to that.
    #
    # Otherwise, forward to `getfield` to return the property of the TableEntry.
    if in(name, names)
        return getproperty(getfield(T, :params), name)
    else
        return getfield(T, name)
    end
end

# Define these for convenience.
_properties(names::Tuple) = (:created, :updated, :data, names...)
_properties(::Type{<:TableEntry{names}}) where {names} = _properties(names)
Base.propertynames(T::TableEntry) = _properties(typeof(T))
_names(::Type{<:TableEntry{names}}) where {names} = names

# Define some methods of promoting a TableEntry
function Base.promote_rule(::Type{TableEntry{N1}}, ::Type{TableEntry{N2}}) where {N1,N2}
    new = Tuple(sort(union(N1, N2)))
    return TableEntry{new}
end

Base.convert(::Type{TableEntry{N}}, x::TableEntry{N}) where {N} = x
function Base.convert(::Type{TableEntry{N}}, x::TableEntry{M}) where {N, M}
    # Get the entries of `N` that are not in `M`
    newnames = Tuple(sort(setdiff(N, M)))

    # Construct a NamedTuple with these names and all values as `missing`
    newentries = NamedTuple{newnames}(ntuple(i -> missing, length(newnames)) )
    newparams = ntsorter(merge(x.params, newentries))

    return TableEntry(
        x.created,
        # We didn't modify any of the contents of the table, just added some `missing` columns.
        x.updated,
        newparams,
        x.data
    )
end

#####
##### DataTable
#####

# The common serialization format for storing all of the benchmark results.
mutable struct DataTable{names}
    entries::Vector{TableEntry{names}}
end
DataTable() = DataTable{()}(TableEntry{()}[])
_properties(::DataTable{names}) where {names} = _properties(names)

# Some helpful entries
Base.getindex(D::DataTable, i::Integer) = DataTable([D.entries[i]])
Base.getindex(D::DataTable, I::Vector) = DataTable([D.entries[i] for i in I])
Base.getindex(D::DataTable, ::Nothing) = DataTable()
Base.getindex(D::DataTable, name::Symbol) = getproperty(Tables.columns(D), name)

function Base.getindex(D::DataTable, params::NamedTuple)
    inds = findall(x -> ismatch(x, params), D.entries)
    return D[inds]
end

Base.deleteat!(D::DataTable, i) = deleteat!(D.entries, i)
Base.length(D::DataTable) = length(D.entries)

Base.propertynames(D::DataTable{names}) where {names} = _properties(names)
function Base.show(io::IO, D::DataTable{names}) where {names}
    if iszero(length(D))
        print(io, "Empty DataTable")
        return nothing
    end
    # Convert all of the data in the table into a matrix for printing.
    #
    # Yes - this could be expensive but who really cares?
    data = [getproperty(e, n) for e in D.entries, n in names]
    header = collect(names)
    pretty_table(io, data, header)
    return nothing
end

# Default the merging to returning the second argument.
#
# We can hijack this method to get data merged correctly for SocketCounters.
datamerge(a, b) = b

function addentry!(
        D::DataTable{N},
        params::NamedTuple,
        data = Dict{String,Any}()
    ) where {N}

    # Sort the parameters by name.
    params = ntsorter(params)
    entry = TableEntry(params, data)

    # Handle the initialization case
    isempty(D.entries) && return DataTable([entry])

    # Otherwise, promote this entry to the entries in the table
    T = promote_type(typeof(entry), eltype(D.entries))
    converted_entry = convert(T, entry)

    @show _names(T)
    @show N

    # Create an entry for these params to get promotion right - then check to see if all
    # of these parameters exist.

    # Check if the parameters we provided are a subset of the colums already in the table.
    # If so, we can search to see if we already have this entry and just add the data to
    # that entry.
    #
    # Otherwise, we must promote the table to the new names and then create a new entry
    # for this.
    if _names(T) == N
        # Filter and see if we already have an existing entry with these params.
        for row in Tables.rows(D)
            # If we have a match, merge the data.
            if ismatch(row, converted_entry)
                println("Found a match!")
                merge!(datamerge, getproperty(row, :data), data)

                # Update the timestamp on the row.
                row.updated = now()
                return D
            end
        end
    end

    # Convert the existing table.
    converted_entries = convert.(T, D.entries)

    # Add the new entry
    push!(converted_entries, convert(T, entry))

    # Return the new table
    return DataTable(converted_entries)
end

#####
##### Tables.jl Interface
#####

Tables.istable(::Type{<:DataTable}) = true

# Extend the `Tables.jl` interface for the DataTable
Tables.rowaccess(::Type{<:DataTable}) = true
Tables.rows(D::DataTable) = D.entries
function Tables.schema(D::DataTable{N}) where {N}
    # Forward to the property names of each entry.
    names = _properties(N)
    types = ntuple(i -> Any, length(names))

    return Tables.Schema(names, types)
end


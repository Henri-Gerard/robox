"""This module implements utility functions."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

function dispatch_index(N::Int, Nprocs::Int)
    r = N % Nprocs
    q = div(N, Nprocs)
    a = cumsum((q+1)*ones(r))
    b = a - q
    if length(a) > 0
        c = cumsum(q*ones(Nprocs-r))+a[end]
    else
        c = cumsum(q*ones(Nprocs-r))
    end
    d = c- (q-1)
    indmax::Array{Int,1} = vcat(a, c)
    indmin::Array{Int,1} = vcat(b, d)
    return indmin, indmax
end

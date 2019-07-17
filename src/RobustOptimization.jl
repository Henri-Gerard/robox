"""This file initialize module robox."""
# Author: Henri Gérard <hgerard.proy@gmail.com>
# License: MIT

__precompile__()

module RobustOptimization

using JuMP, StatsBase
using Distributed
using DelimitedFiles
using LinearAlgebra

export DRO_opt, normal_opt, KL_opt, run_algo,
        error_loss, error_prime, equivalent,
        fconstraint, subgradient, typeof_constraint,
        x_in_inter, aux_proj, QR, proj_pi,
        GeneralConstraint, KLConstraint, DROConstraint, PositiveConstraint, EntropicConstraint,
        LinearRegression, LogisticRegression,
        Parallel, Sequential,
        OptParams, ProjParams,
        RobustModel,
        DRO_opt, normal_opt, KL_opt,
        read_data_libsvm, getsolution, pred, initialize, init_proj, create_data,
        dispatch_index, proj_pi, algo_proj_bis, algo_proj, run_algo_bis, positive_rate


include("objects.jl")
include("projection.jl")
include("ipsolve.jl")
include("preprocess.jl")
include("postprocess.jl")
include("constraint_functions.jl")
include("error_functions.jl")
include("utils.jl")

end

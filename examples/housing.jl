println("Start")
# Naddprocs = 1
# if Naddprocs > 1
#     addprocs(Naddprocs)
#
#     println("Load module")
#     @everywhere push!(LOAD_PATH, ".")
#     @everywhere using RobustOptimization, ScikitLearn
#     using Ipopt, PyPlot
#     @everywhere @sk_import model_selection: train_test_split
#     @everywhere @sk_import metrics: r2_score
#
#     @everywhere nbfeatures = 13
#     solver = IpoptSolver(print_level = 2)
#
#     println("Create date")
#     @everywhere df_train, df_test = create_data("housing_scale", nbfeatures, 0.6)
# else
println("Load module")
push!(LOAD_PATH, ".")
using RobustOptimization
using ScikitLearn
using Ipopt, PyPlot
@sk_import model_selection: train_test_split
@sk_import metrics: r2_score

nbfeatures = 13
solver = IpoptSolver(print_level = 2)

println("Create data")
df_train, df_test = create_data("housing_scale", nbfeatures, 0.1, train_test_split)


println("Define model")
N = size(df_train)[1]
ϵ = 0.1
verbosity = 100
itmax = 5000
sample = 4

ambiguity = "wasserstein"
robustModel = RobustModel(N, nbfeatures, ϵ, ambiguity, LinearRegression())
α = 1/norm(robustModel.descent_direction)
projParams = ProjParams(Int(1e6), 1e-5, sample, para_proj=Sequential(), para_inter=Sequential())
optParams = OptParams(itmax, 1e-7, α, verbosity = verbosity);



println("Solve normal")
xnormal = normal_opt(df_train, solver)
y_pred = pred(df_test, xnormal, robustModel.regressionModel)
y_true = df_test[:,end]
println(" ")
println(" ")
println("xnormal = ", xnormal)
println("rnormal = ", r2_score(y_true, y_pred))
println(" ")
println(" ")



println("Solve wasserstein")
x0 = init_proj(df_train, robustModel, projParams)
println("fin_init")
xalg, yalg, dmwas, mem, mini = run_algo(x0, df_train, robustModel, optParams, projParams)
xrobust1 = getsolution(xalg, ambiguity, nbfeatures)
xrobust2 = getsolution(mem, ambiguity, nbfeatures)
y_pred1 = pred(df_test, xrobust1, robustModel.regressionModel)
y_pred2 = pred(df_test, xrobust2, robustModel.regressionModel)
y_true = df_test[:,end]
r1 = r2_score(y_true, y_pred1)
r2 = r2_score(y_true, y_pred2)
if r1 > r2
    println(" ")
    println(" ")
    println("xrobwas = ", xrobust1)
    println("r1 = ", r2_score(y_true, y_pred1))
    println(" ")
    println(" ")
else
    println(" ")
    println(" ")
    println("xrobwas = ", xrobust2)
    println("r2 = ", r2_score(y_true, y_pred2))
    println(" ")
    println(" ")
end
println("Fin wasserstein")


ambiguity = "KLdivergence"
robustModel = RobustModel(N, nbfeatures, ϵ, ambiguity, LinearRegression())
α = 1/norm(robustModel.descent_direction)
projParams = ProjParams(Int(1e6), 1e-5, sample, para_proj=Sequential(), para_inter=Sequential())
optParams = OptParams(itmax, 1e-7, α, verbosity = verbosity);

println("Solve KL")
x0 = init_proj(df_train, robustModel, projParams)
println("fin_init")
xalg, yalg, dmKL, mem, mini = run_algo(x0, df_train, robustModel, optParams, projParams)
xrobust1 = getsolution(xalg, ambiguity, nbfeatures)
xrobust2 = getsolution(mem, ambiguity, nbfeatures)
y_pred1 = pred(df_test, xrobust1, robustModel.regressionModel)
y_pred2 = pred(df_test, xrobust2, robustModel.regressionModel)
y_true = df_test[:,end]
r1 = r2_score(y_true, y_pred1)
r2 = r2_score(y_true, y_pred2)
if r1 > r2
    println(" ")
    println(" ")
    println("xrobKL = ", xrobust1)
    println("r1 = ", r2_score(y_true, y_pred1))
    println(" ")
    println(" ")
else
    println(" ")
    println(" ")
    println("xrobKl = ", xrobust2)
    println("r2 = ", r2_score(y_true, y_pred2))
    println(" ")
    println(" ")
end

println("Fin KL")

println("dmwas = ", dmwas)
println("dmKl = ", dmKL)

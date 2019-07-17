push!(LOAD_PATH, "../src")
using RobustOptimization
using Ipopt, JuMP, PyPlot
using ScikitLearn
using LinearAlgebra
@sk_import model_selection: train_test_split
@sk_import metrics: roc_curve
@sk_import metrics: roc_auc_score
@sk_import metrics: accuracy_score
@sk_import metrics: classification_report
path_train = "data/ionosphere_scale"
nbfeatures = 34
data_train = read_data_libsvm(path_train, nbfeatures);
# xtr, xte, ytr, yte = train_test_split(data_train[:,1:end-1], data_train[:,end], train_size = 0.15)
# ab_train = hcat(xtr, ytr)
# aux_train = ab_train[ab_train[:,end] .< 0,:]
# ll_train = vcat(aux_train,data_train[data_train[:,end].>0,:])

rs = 75
xtr, xte, ytr, yte = train_test_split(data_train[:,1:end-1], data_train[:,end], train_size = 0.6, random_state = rs)
df_train = hcat(xtr, ytr)
# low_train = hcat(xtr, ytr)
df_aux= hcat(xte, yte);

# df_train, df_aux = create_data("diabetes_scale", nbfeatures, 0.2, train_test_split)
solver = with_optimizer(Ipopt.Optimizer, print_level=2)
N = size(df_train)[1]
ϵ = 0.05
verbosity = 500
itmax = 100
sample = N

ambiguity = "KLdivergence"
robustModel = RobustModel(N, nbfeatures, ϵ, ambiguity, LogisticRegression())
α = 1/norm(robustModel.descent_direction)
# α = 0.1
projParams = ProjParams(Int(40), 1e-5, sample, para_proj=Sequential(), para_inter=Sequential())
optParams = OptParams(itmax, 1e-5, α, verbosity = verbosity);

xnormal = normal_opt(df_train, solver, robustModel.regressionModel)
fpr, tpr, thresholds = roc_curve(df_aux[:,end], positive_rate(xnormal,df_aux))

e_tab = [0.0001,0.0005,0.0006,0.0007,0.0008,0.0009,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.01,0.05,0.1,0.5]
# e_tab = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.05,0.1,0.5,1.0]

dm_tab = []
time_tab = []
xr1_tab = []
xr2_tab = []
fpr1_tab = []
tpr1_tab = []
fpr2_tab = []
tpr2_tab = []
for ϵ in e_tab
    println(" ")
    println("espilon = ", ϵ)
    println(" ")
    robustModel = RobustModel(N, nbfeatures, ϵ, ambiguity, LogisticRegression())
    x0 = initialize(df_train, robustModel, KLConstraint())
    xalg, yalg, dm, mem, mini, t_iter = run_algo(x0, df_train, robustModel, optParams, projParams)
    xrobust1 = getsolution(xalg, ambiguity, nbfeatures)
    xrobust2 = getsolution(mem, ambiguity, nbfeatures)
    fpr1, tpr1, thresholds = roc_curve(df_aux[:,end], positive_rate(xrobust1,df_aux))
    fpr2, tpr2, thresholds = roc_curve(df_aux[:,end], positive_rate(xrobust2,df_aux))
    push!(time_tab, t_iter)
    push!(dm_tab, dm)
    push!(xr1_tab, xrobust1)
    push!(xr2_tab, xrobust2)
    push!(fpr1_tab, fpr1)
    push!(tpr1_tab, tpr1)
    push!(fpr2_tab, fpr2)
    push!(tpr2_tab, tpr2)
end

figure()
for i in 1:(size(e_tab)[1]-1)
    plot(dm_tab[i][1:end].-dm_tab[i][end], label="ϵ = $(e_tab[i])")
end
legend()

figure()
for i in 1:(size(e_tab)[1]-1)
    plot((dm_tab[i]), label="ϵ = $(e_tab[i])")
    yscale("log")
end
legend()

figure()
for i in 1:(size(e_tab)[1]-1)
    plot(cumsum(time_tab[1]), dm_tab[1], label="ϵ = $(e_tab[i])")
    yscale("log")
end
legend(loc=1)

figure()
plot(fpr,tpr, color="black", label="LR")
for i in 1:(size(e_tab)[1]-1)
    plot(fpr1_tab[i], tpr1_tab[i], label="ϵ = $(e_tab[i])")
end
legend()

figure()
plot(fpr,tpr, color="black")
for i in 1:14
    plot(fpr2_tab[i], tpr2_tab[i])
end

plt_AUC = []
print(roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
push!(plt_AUC,roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
print("  ")
println(accuracy_score(df_aux[:,end], pred(df_aux, xnormal, LogisticRegression())))
for i in 1:(size(e_tab)[1])
    push!(plt_AUC, roc_auc_score(df_aux[:,end], positive_rate(xr1_tab[i],df_aux)))
    print(roc_auc_score(df_aux[:,end], positive_rate(xr1_tab[i],df_aux)))
    print("  ")
    println(accuracy_score(df_aux[:,end], pred(df_aux, xr1_tab[i], LogisticRegression())))
end
figure()
plot(vcat([0],e_tab),plt_AUC, marker="*")
xscale("log")

plt_AUC = []
print(roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
push!(plt_AUC,roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
print("  ")
println(accuracy_score(df_aux[:,end], pred(df_aux, xnormal, LogisticRegression())))
for i in 1:(size(e_tab)[1])
    push!(plt_AUC, roc_auc_score(df_aux[:,end], positive_rate(xr1_tab[i],df_aux)))
    print(roc_auc_score(df_aux[:,end], positive_rate(xr2_tab[i],df_aux)))
    print("  ")
    println(accuracy_score(df_aux[:,end], pred(df_aux, xr2_tab[i], LogisticRegression())))
end
figure()
plot(vcat([0],e_tab),plt_AUC, marker="*")
xscale("log")

# @everywhere df_train, df_aux = create_data("diabetes_scale", nbfeatures, 0.7, train_test_split)
solver = with_optimizer(Ipopt.Optimizer, print_level=2)
N = size(df_train)[1]
ϵ = 0.05
verbosity = 1
itmax = 50
sample = 1500

ambiguity = "wasserstein"
robustModel = RobustModel(N, nbfeatures, ϵ, ambiguity, LogisticRegression())
α = 1/norm(robustModel.descent_direction)
# α = 1.0
projParams = ProjParams(Int(200), 1e-5, sample, para_proj=Sequential(), para_inter=Sequential())
optParams = OptParams(itmax, 1e-5, α, verbosity = verbosity);

xnormal = normal_opt(df_train, solver, robustModel.regressionModel)
fpr, tpr, thresholds = roc_curve(df_aux[:,end], positive_rate(xnormal,df_aux))

# e_tab_was = [0.1]
# e_tab_was = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
e_tab_was = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0]

mem_was = []
dm_was = []
time_was = []
mini_was = []
xr1_was = []
xr2_was = []
fpr1_was = []
tpr1_was = []
fpr2_was = []
tpr2_was = []
for ϵ in e_tab_was
# for ϵ in [0.001,0.005,0.01]
    println(" ")
    println("espilon = ", ϵ)
    println(" ")
    robustModel = RobustModel(N, nbfeatures, ϵ, ambiguity, LogisticRegression())
    x0 = init_proj(df_train, robustModel, projParams);
    xalg, yalg, dm, mem, mini, t_was = run_algo(x0, df_train, robustModel, optParams, projParams)
    xrobust1 = getsolution(xalg, ambiguity, nbfeatures)
    xrobust2 = getsolution(mem, ambiguity, nbfeatures)
    fpr1, tpr1, thresholds = roc_curve(df_aux[:,end], positive_rate(xrobust1,df_aux))
    fpr2, tpr2, thresholds = roc_curve(df_aux[:,end], positive_rate(xrobust2,df_aux))
    push!(dm_was, dm)
    push!(time_was, dm)
    push!(mini_was, mini)
    push!(mem_was, mem)
    push!(xr1_was, xrobust1)
    push!(xr2_was, xalg)
    push!(fpr1_was, fpr1)
    push!(tpr1_was, tpr1)
    push!(fpr2_was, fpr2)
    push!(tpr2_was, tpr2)
end

figure()
for i in 1:size(e_tab_was)[1]
    plot(dm_was[i], label="ϵ = $(e_tab[i])")
end
legend()

figure()
for i in 1:(size(e_tab_was)[1]-2)
    plot(dm_was[i][1:end].-minimum(dm_was[i]), label="ϵ = $(e_tab_was[i])")
    yscale("log")
end
# plot(dm_was[11][1:end].-minimum(dm_was[11]), label="ϵ = $(e_tab_was[end-1])")
# plot(dm_was[end][1:end].-minimum(dm_was[end]), label="ϵ = $(e_tab_was[end])")
yscale("log")
xlabel("Iterations")
legend(loc = 1, ncol=2)

figure()
for i in 1:size(e_tab_was)[1]
    plot(cumsum(time_was[i][1:end]), dm_was[i][1:end].-mini_was[i], label="ϵ = $(e_tab[i])")
    yscale("log")
end
legend()

figure()
plot(fpr,tpr, color="black", label="LR")
for i in 1:size(e_tab_was)[1]
    plot(fpr1_was[i], tpr1_was[i], label="ϵ = $(e_tab[i])")
end
legend()

figure()
plot(fpr,tpr, color="black")
for i in 1:size(e_tab_was)[1]
    plot(fpr2_was[i], tpr2_was[i])
end

plt_was = []
print(roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
push!(plt_was, roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
print("  ")
println(accuracy_score(df_aux[:,end], pred(df_aux, xnormal, LogisticRegression())))
for i in 1:size(e_tab_was)[1]
    print(roc_auc_score(df_aux[:,end], positive_rate(xr1_was[i],df_aux)))
    push!(plt_was, roc_auc_score(df_aux[:,end], positive_rate(xr1_was[i],df_aux)))
    print("  ")
    println(accuracy_score(df_aux[:,end], pred(df_aux, xr1_was[i], LogisticRegression())))
end
figure()
plot(vcat([0],e_tab_was),plt_was, marker="*")
xscale("log")


print(roc_auc_score(df_aux[:,end], positive_rate(xnormal,df_aux)))
print("  ")
println(accuracy_score(df_aux[:,end], pred(df_aux, xnormal, LogisticRegression())))
for i in 1:21
    print(roc_auc_score(df_aux[:,end], positive_rate(xr2_was[i],df_aux)))
    print("  ")
    println(accuracy_score(df_aux[:,end], pred(df_aux, xr2_was[i], LogisticRegression())))
end

println(classification_report(df_aux[:,end], pred(df_aux, xnormal, LogisticRegression())))
println(classification_report(df_aux[:,end], pred(df_aux, xr1_was[3], LogisticRegression())))

# open("io-log-was-good.txt", "w") do f
#     write(f, "xnormal = ")
#     write(f, "$xnormal \n")
# #     write(f, "fpr = ")
# #     write(f, "$fpr \n")
# #     write(f, "tpr = ")
# #     write(f, "$tpr \n")
# #     write(f, "dm_tab = ")
# #     write(f, "$dm_tab \n")
# #     write(f, "time_tab = ")
# #     write(f, "$time_tab \n")
# #     write(f, "xr1_tab = ")
# #     write(f, "$xr1_tab \n")
# #     write(f, "fpr1_tab = ")
# #     write(f, "$fpr1_tab \n")
# #     write(f, "tpr1_tab = ")
# #     write(f, "$tpr1_tab \n")
#     write(f, "dm_was = ")
#     write(f, "$dm_was \n")
#     write(f, "time_was = ")
#     write(f, "$time_was \n")
#     write(f, "xr1_was = ")
#     write(f, "$xr1_was \n")
#     write(f, "fpr1_was = ")
#     write(f, "$fpr1_was \n")
#     write(f, "tpr1_was = ")
#     write(f, "$tpr1_was \n")
# end

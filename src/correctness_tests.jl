###############################
#
# Correctness testing
#
################################

# Standard POMDP
# All values calculated using POMDPs on Julia 1.0
#
# Tiger Problem -------------------------------------------------------
# TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)
# tolerance = 0.001
#
# SARSOP
# Expected: 19.372
# Simulator: Sim ETR 19.3611, Sim ETR 95% CI (17.48, 21.241)
# Evaluator: Eval ETR 19.4831, Sim ETR 95% CI (19.2147, 19.7516)
# Policy: POMDPXFiles.POMDPAlphas([-81.5975 3.01448 … 28.4025 19.3711; 28.4025 24.6954 … -81.5975 19.3711], [2, 0, 0, 1, 0])
# alphasref_tiger = [[-81.5975, 28.4025],
#         [3.01448, 24.6954],
#         [24.6954, 3.01452],
#         [28.4025, -81.5975],
#         [19.3711, 19.3711]]
#
# Baby Problem -------------------------------------------------------
# BabyPOMDP(-5.0, -10.0, 0.1, 0.8, 0.1, 0.9)
# tolerance = 0.001
#
# SARSOP:
# Expected: -16.305
# Simulator: Sim ETR -16.505, Sim ETR 95% CI (-17.075, -15.935)
# Evaluator: Eval ETR -16.384, Sim ETR 95% CI (-16.749, -16.019)
# Policy: POMDPXFiles.POMDPAlphas([-19.6749 -16.3055; -29.6749 -38.2512], [1, 0])
# alphasref_baby = [[-19.6749, -29.6749], [-16.3055, -38.2512]]

# compare RPOMDP to POMDP
using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
using Plots; gr()
const RPBVI = RobustValueIteration

# PBVI POMDPs
srand(8473272)
bs = [[b, 1-b] for b in 0.0:0.01:1.0]
prob = TigerPOMDP(0.95)
prob2 = Baby2POMDP(-5.0, -10.0, 0.9)
sr = RPBVISolver(beliefpoints = bs, max_iterations = 200)
sr2 = RPBVISolver(beliefpoints = bs, max_iterations = 200)
polr = RobustValueIteration.solve(sr, prob)
polr2 = RobustValueIteration.solve(sr2, prob2)
bur = updater(polr)
bur2 = updater(polr2)
value(polr, [0.5,0.5])
value(polr2, [0.0, 1.0])
N = 1_000
N2 = 1_000
simvals = [discounted_reward(simulate(HistoryRecorder(max_steps=1_000),
                                                prob, polr, bur)) for i = 1:N]
m = mean(simvals)
s = std(simvals)
tstar = 1.962
ci = (m - tstar * s / sqrt(N), m + tstar * s / sqrt(N))

simvals2 = [discounted_reward(simulate(HistoryRecorder(max_steps=100),
                                                prob2, polr2, bur2)) for i = 1:N2]
m2 = mean(simvals2)
s2 = std(simvals2)
tstar = 1.962
ci2 = (m2 - tstar * s2 / sqrt(N2), m2 + tstar * s2 / sqrt(N2))

using Plots; gr()
plotvals0_tiger = [maximum(dot(polr.alphas[i], [1-b, b]) for i = 1:length(polr.alphas)) for b = 0.0:0.01:1.0]
alphasref_tiger = [[-81.5975, 28.4025],
        [3.01448, 24.6954],
        [24.6954, 3.01452],
        [28.4025, -81.5975],
        [19.3711, 19.3711]]

valsref_tiger = [maximum(dot(alphasref_tiger[i], [b, 1-b]) for i = 1:length(alphasref_tiger))
        for b = 0.0:0.01:1.0]

x = collect(0:0.01:1.0)
p_tiger = Plots.plot(x, plotvals0_tiger, xticks = 0:0.1:1,
        legend = :top, label = "PBVI",
        # title = "Tiger POMDP Value Function",
        xlab = "Belief, P(State = Tiger Left)",
        ylab = "Expected Total Discounted Reward",
        line = (4, :solid, :red), alpha = 0.5)
plot!(x, valsref_tiger,
    label = "SARSOP", line = :dash, alpha = 1, linewidth = 2,
    linecolor = :blue)
fn = "value_function_tiger_2.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_tiger, path)

plotvals0_baby = [maximum(dot(polr2.alphas[i], [b, 1-b]) for i = 1:length(polr2.alphas)) for b = 0.0:0.01:1.0]
alphasref_baby = [[-19.6749, -29.6749], [-16.3055, -38.2512]]
valsref_baby = [maximum(dot(alphasref_baby[i], [1-b, b]) for i = 1:length(alphasref_baby))
        for b = 0.0:0.01:1.0]

x = collect(0:0.01:1.0)
p_baby = Plots.plot(x, plotvals0_baby, xticks = 0:0.1:1,
        legend = :topright, label = "PBVI",
        # title = "Baby POMDP Value Function",
        xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        line = (4, :solid, :red, 0.5))
plot!(x, valsref_baby, label = "SARSOP",
    line = (2, :dash, :blue, 1.0))
fn = "value_function_baby_2.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_baby, path)


#########################################
# RPBVI Robust POMDPs Tiger
# unc size = 0.001, 0.01, 0.05
srand(8473272)
nr = 5
bs = fill([[b, 1-b] for b in 0.0:0.05:1.0], nr)
ambiguity = [0.001, 0.01, 0.05, 0.1, 0.2]
maxiter = [100, 100, 100, 100, 100]
nreps = [1, 1, 500, 1, 1]
maxstep = [100, 100, 100, 100, 100]
policyvalue_tiger = Vector{Float64}(nr)
simvals_tiger = Vector{Vector{Float64}}(nr)
simps_tiger = Vector{Vector{Float64}}(nr)
simvals_tiger_nom = Vector{Vector{Float64}}(nr)
simps_tiger_nom = Vector{Vector{Float64}}(nr)
m_tiger = Vector{Float64}(nr)
m_nom_tiger = Vector{Float64}(nr)
s_tiger = Vector{Float64}(nr)
ci_tiger = Vector{Tuple{Float64, Float64}}(nr)
ci_nom_tiger = Vector{Tuple{Float64, Float64}}(nr)
plotvals_tiger = Vector{Vector{Float64}}(nr)

# plot value
for i = 1:nr
    println("Starting loop $i...")
    prob = TigerRPOMDP(0.95, ambiguity[i])
    sr = RPBVISolver(beliefpoints = bs[i], max_iterations = maxiter[i])
    println("Calculating policy...")
    polr = RobustValueIteration.solve(sr, prob)
    plotvals_tiger[i] = [maximum(dot(polr.alphas[j], [1-b, b]) for j = 1:length(polr.alphas))
        for b = 0.0:0.01:1.0]
end

# function meanci(data::Vector{Float64})
#     n = length(data)
#     m = mean(data)
#     s = std(data)
#     tstar = 1.962
#     hw = tstar * s / sqrt(n)
#     m, (m - hw, m + hw)
# end
#
# nreps[1] = 500
# uncsize = 0.05
# prob_nom = TigerPOMDP(0.95)
# solver_nom = RPBVISolver(beliefpoints = bs[1], max_iterations = maxiter[1])
# pol_nom = RPBVI.solve(solver_nom, prob_nom)
# bu_nom = updater(pol_nom)
# prob_r_nom = SimpleTigerRPOMDP(0.95, uncsize)
# pol_r_nom = RPBVI.solve(solver_nom, prob_r_nom)
# bu_r_nom = updater(pol_r_nom)
# simulator = RolloutSimulator(max_steps = maxstep[1])
# simvals_nn = Array{Float64}(nr, nreps[1],2)
# simvals_nr = Array{Float64}(nr, nreps[1],2)
# simvals_rn = Array{Float64}(nr, nreps[1],2)
# simvals_rr = Array{Float64}(nr, nreps[1],2)
# for j = 1:nreps[1]
#     (j % 10 == 0) && print("\rRep $j")
#     # nominal policy against nominal dynamics
#     simvals_nn[1,j,1], simvals_nn[1,j,2] = simulate(simulator,
#         prob_nom, pol_nom, bu_nom)
#     # nominal policy against worst-case dynamics
#     simvals_nr[1,j,1], simvals_nr[1,j,2]  = simulate_worst(simulator,
#         prob_r_nom, pol_nom, bu_nom, pol_r_nom.alphas)
#     # robust policy against nominal dynamics
#     simvals_rn[1,j,1], simvals_rn[1,j,2] = simulate(simulator,
#         prob_nom, pol_r_nom, bu_r_nom)
#     # robust policy against worst-case dynamics
#     simvals_rr[1,j,1], simvals_rr[1,j,2]  = simulate_worst(simulator,
#         prob_r_nom, pol_r_nom, bu_r_nom, pol_r_nom.alphas)
# end
# value(pol_nom, [0.5, 0.5])
# value(pol_r_nom, [0.5, 0.5])
# m_nn, ci_nn = meanci(simvals_nn[1,:,1])
# m_nr, ci_nr = meanci(simvals_nr[1,:,1])
# m_rn, ci_rn = meanci(simvals_rn[1,:,1])
# m_rr, ci_rr = meanci(simvals_rr[1,:,1])
#
#
# # sim value
# srand(8473272)
# for i = 1:1
#     println("Loop $i")
#     prob = TigerRPOMDP(0.95, ambiguity[i])
#     sr = RPBVISolver(beliefpoints = bs[i], max_iterations = maxiter[i])
#     println("Calculating policy...")
#     polr = RobustValueIteration.solve(sr, prob)
#     bur = updater(polr)
#     policyvalue_tiger[i] = value(polr, [0.5,0.5])
#     println("Simulating value...")
#     simvals_tiger_temp = Vector{}(nreps[i])
#     simps_tiger_temp = Vector{}(nreps[i])
#     simvals_tiger_nom_temp = Vector{}(nreps[i])
#     simps_tiger_nom_temp = Vector{}(nreps[i])
#     simulator = RolloutSimulator(max_steps = maxstep[i])
#     for j = 1:nreps[i]
#         (j % 10 == 0) && print("\rRep $j")
#         simvals_tiger_temp[j], simps_tiger_temp[j]  = simulate_worst(simulator,
#             prob, polr, bur, polr.alphas)
#         simvals_tiger_nom_temp[j], simps_tiger_nom_temp[j]  = simulate_worst(simulator,
#             prob, pol_nom, bu_nom, polr.alphas)
#     end
#     simvals_tiger[i] = simvals_tiger_temp
#     simps_tiger[i] = simps_tiger_temp
#     simvals_tiger_nom[i] = simvals_tiger_nom_temp
#     simps_tiger_nom[i] = simps_tiger_nom_temp
#     m_tiger[i] = mean(simvals_tiger[i])
#     m_nom_tiger[i] = mean(simvals_tiger_nom[i])
#     s_tiger[i] = std(simvals_tiger[i])
#     tstar = 1.962
#     ci_tiger[i] = (m_tiger[i] - tstar * s_tiger[i] / sqrt(nreps[i]), m_tiger[i] + tstar * s_tiger[i] / sqrt(nreps[i]))
#     ci_nom_tiger[i] = (m_nom_tiger[i] - tstar * std(simvals_tiger_nom[i]) / sqrt(nreps[i]),
#         m_nom_tiger[i] + tstar * std(simvals_tiger_nom[i]) / sqrt(nreps[i]))
# end

# reference values
alphasref_tiger = [[-81.5975, 28.4025],
        [3.01448, 24.6954],
        [24.6954, 3.01452],
        [28.4025, -81.5975],
        [19.3711, 19.3711]]
valsref_tiger = [maximum(dot(alphasref_tiger[i], [b, 1-b]) for i = 1:length(alphasref_tiger))
        for b = 0.0:0.01:1.0]

x = collect(0:0.01:1.0)
p_tiger_robust = plot(x, [valsref_tiger, plotvals_tiger],
        xticks = 0:0.1:1, label = "SARSOP",
        xlab = "Belief, P(State = Tiger Left)",
        ylab = "Expected Total Discounted Reward",
        legend = :none,
        linestyle = [:dot :solid :solid :solid :solid :solid],
        linewidth = [4 2 2 2 2 2],
        linealpha = [1 0.9 0.7 0.5 0.3 0.1],
        linecolor = [:blue :red :red :red :red :red])
# plot!(x, plotvals0_tiger, line = (2, :red, 1.0), label = "PBVI")
tigerrobustlabels = ["SARSOP", "RPBVI: 0.001", "RPBVI: 0.01", "RPBVI: 0.05", "RPBVI: 0.1", "RPBVI: 0.2"]
l = @layout [a b{0.2w}]
p1 = p_tiger_robust
p2 = plot(x, [valsref_tiger, plotvals_tiger],
        linestyle = [:dot :solid :solid :solid :solid :solid],
        linewidth = [4 2 2 2 2 2],
        linealpha = [1 0.9 0.7 0.5 0.3 0.1],
        linecolor = [:blue :red :red :red :red :red],
        label=tigerrobustlabels, grid=false, xlims=(20,3), showaxis=false)
p_tiger_robust_out = plot(p1,p2,layout=l)
fn = "value_function_tiger_robust_4.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_tiger_robust_out, path)

# RPBVI Robust POMDPs Baby
# unc size = 0.001, 0.1
srand(8473272)
nr = 5
bs = [[[b, 1-b] for b in 0.0:0.05:1.0],
        [[b, 1-b] for b in 0.0:0.05:1.0],
        [[b, 1-b] for b in 0.0:0.05:1.0],
        [[b, 1-b] for b in 0.0:0.05:1.0],
        [[b, 1-b] for b in 0.0:0.05:1.0]]
ambiguity = [0.001, 0.01, 0.05, 0.1, 0.2]
maxiter = [100, 100, 100, 100, 100]
nreps = [1, 1, 500, 1, 1]
maxstep = [100, 100, 100, 100, 100]
policy_baby = Vector{}(nr)
policyvalue_baby = Vector{}(nr)
simvals_baby = Vector{}(nr)
m_baby = Vector{}(nr)
s_baby = Vector{}(nr)
ci_baby = Vector{}(nr)
plotvals_baby = Vector{}(nr)

# plot value
for i = 1:nr
    println("Starting loop $i...")
    prob = Baby2RPOMDP(-5.0, -10.0, 0.9, ambiguity[i])
    sr = RPBVISolver(beliefpoints = bs[i], max_iterations = maxiter[i])
    println("Calculating policy...")
    policy_baby[i] = RobustValueIteration.solve(sr, prob)
    policyvalue_baby[i] = value(policy_baby[i], [0.0,1.0])
    plotvals_baby[i] = [maximum(dot(policy_baby[i].alphas[j], [b, 1-b]) for j = 1:length(policy_baby[i].alphas))
        for b = 0.0:0.01:1.0]
end

# reference values
alphasref_baby = [[-19.6749, -29.6749], [-16.3055, -38.2512]]
valsref_baby = [maximum(dot(alphasref_baby[i], [1-b, b]) for i = 1:length(alphasref_baby))
        for b = 0.0:0.01:1.0]

x = collect(0:0.01:1.0)
p_baby_robust = plot(x, valsref_baby, xticks = 0:0.1:1, label = "SARSOP",
        # title = "Baby POMDP Value Function",
        xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        legend = :bottomright,
        color = :blue,
        line = :dot,
        linealpha = 1.0,
        linewidth = 4)
# plot!(x, plotvals0_baby, line = (2, :red, 1.0), label = "PBVI")
plot!(x, plotvals_baby[1], line = (2, :red, 0.9), label = "RPBVI: 0.001")
plot!(x, plotvals_baby[2], line = (2, :red, 0.7), label = "RPBVI: 0.01")
plot!(x, plotvals_baby[3], line = (2, :red, 0.5), label = "RPBVI: 0.05")
plot!(x, plotvals_baby[4], line = (2, :red, 0.3), label = "RPBVI: 0.1")
plot!(x, plotvals_baby[5], line = (2, :red, 0.1), label = "RPBVI: 0.2")
fn = "value_function_baby_robust_3.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_baby_robust, path)

p_baby_robust = plot(x, [valsref_baby, plotvals_baby],
        xticks = 0:0.1:1, label = "SARSOP",
        xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        legend = :none,
        linestyle = [:dot :solid :solid :solid :solid :solid],
        linewidth = [4 2 2 2 2 2],
        linealpha = [1 0.9 0.7 0.5 0.3 0.1],
        linecolor = [:blue :red :red :red :red :red])
# plot!(x, plotvals0_tiger, line = (2, :red, 1.0), label = "PBVI")
babyrobustlabels = ["SARSOP", "RPBVI: 0.001", "RPBVI: 0.01", "RPBVI: 0.05", "RPBVI: 0.1", "RPBVI: 0.2"]
l = @layout [a b{0.2w}]
p1 = p_baby_robust
p2 = plot(x, [valsref_baby, plotvals_baby],
        linestyle = [:dot :solid :solid :solid :solid :solid],
        linewidth = [4 2 2 2 2 2],
        linealpha = [1 0.9 0.7 0.5 0.3 0.1],
        linecolor = [:blue :red :red :red :red :red],
        label=babyrobustlabels, grid=false, xlims=(20,3), showaxis=false)
p_baby_robust_out = plot(p1,p2,layout=l)
fn = "value_function_baby_robust_4.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_baby_robust_out, path)


############################################
#
# Baby2RPOMDP Sim Value
#
############################################
using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
using Plots; gr()
const RPBVI = RobustValueIteration

# RPBVI Robust POMDPs Baby2
# unc size = 0.001, 0.01, 0.05
srand(8473272)
nr = 5
bs = fill([[b, 1-b] for b in 0.0:0.05:1.0], nr)
ambiguity = [0.001, 0.01, 0.05, 0.1, 0.2]
maxiter = [100, 100, 100, 100, 100]
nreps = [50, 50, 50, 50, 50]
maxstep = [100, 100, 100, 100, 100]

function meanci(data::Vector{Float64})
    n = length(data)
    m = mean(data)
    s = std(data)
    tstar = 1.962
    hw = tstar * s / sqrt(n)
    m, (m - hw, m + hw)
end

nreps[1] = 200
uncsize = 0.4
rfeed = -15.0
prob_nom = Baby2POMDP(rfeed, -10.0, 0.9)
# prob_nom = TigerPOMDP(0.95)
solver_nom = RPBVISolver(beliefpoints = bs[1], max_iterations = maxiter[1])
pol_nom = RPBVI.solve(solver_nom, prob_nom)
bu_nom = updater(pol_nom)
prob_r_nom = SimpleBaby2RPOMDP(rfeed, -10.0, 0.9, uncsize)
pol_r_nom = RPBVI.solve(solver_nom, prob_r_nom)
@show pol_nom.action_map
@show pol_r_nom.action_map
# prob_r_nom = SimpleTigerRPOMDP(0.95, uncsize)
# prob_r_nom = SimpleTigerRPOMDP(0.95, uncsize)
bu_r_nom = updater(pol_r_nom)
simulator = RolloutSimulator(max_steps = maxstep[1])
simvals_nn = Array{Float64}(nr, nreps[1],2)
simvals_nr = Array{Float64}(nr, nreps[1],2)
simvals_rn = Array{Float64}(nr, nreps[1],2)
simvals_rr = Array{Float64}(nr, nreps[1],2)
for j = 1:nreps[1]
    (j % 10 == 0) && print("\rRep $j")
    # nominal policy against nominal dynamics
    simvals_nn[1,j,1], simvals_nn[1,j,2] = simulate(simulator,
        prob_nom, pol_nom, bu_nom)
    # nominal policy against worst-case dynamics
    simvals_nr[1,j,1], simvals_nr[1,j,2]  = simulate_worst(simulator,
        prob_r_nom, pol_nom, bu_nom, pol_r_nom.alphas)
    # robust policy against nominal dynamics
    simvals_rn[1,j,1], simvals_rn[1,j,2] = simulate(simulator,
        prob_nom, pol_r_nom, bu_r_nom)
    # robust policy against worst-case dynamics
    simvals_rr[1,j,1], simvals_rr[1,j,2]  = simulate_worst(simulator,
        prob_r_nom, pol_r_nom, bu_r_nom, pol_r_nom.alphas)
end
value(pol_nom, [0.0, 1.0])
value(pol_r_nom, [0.0, 1.0])
value(pol_nom, [0.5, 0.5])
value(pol_r_nom, [0.5, 0.5])
m_nn, ci_nn = meanci(simvals_nn[1,:,1])
m_nr, ci_nr = meanci(simvals_nr[1,:,1])
m_rn, ci_rn = meanci(simvals_rn[1,:,1])
m_rr, ci_rr = meanci(simvals_rr[1,:,1])

mp_nn, cip_nn = meanci(simvals_nn[1,:,2])
mp_nr, cip_nr = meanci(simvals_nr[1,:,2])
mp_rn, cip_rn = meanci(simvals_rn[1,:,2])
mp_rr, cip_rr = meanci(simvals_rr[1,:,2])



############################################
#
# RockDiagnosis Sim Value
#
############################################
using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
using SimpleProbabilitySets
# using Plots; gr()
const RPBVI = RobustValueIteration

# RPBVI Robust POMDPs Baby2
# unc size = 0.001, 0.01, 0.05
srand(8473272)
nr = 5
# nb = 30
# bs1 = [vcat(psample(zeros(2), ones(2)),zeros(2)) for i = 1:nb]
# bs2 = [vcat(zeros(2), psample(zeros(2), ones(2))) for i = 1:nb]
# bs0 = vcat(bs1, bs2)
bdelt = 0.05
nb2 = 10
bs1 = [vcat([b, 1-b],zeros(2)) for b in 0.0:bdelt:1.0]
bs2 = [vcat(zeros(2), [b, 1-b]) for b in 0.0:bdelt:1.0]
bs3 = [psample(zeros(4), ones(4)) for i = 1:nb2]
bs0 = vcat(bs1, bs2, bs3)
bs = fill(bs0, nr)
ambiguity = [0.001, 0.1, 0.2, 0.3, 0.4]
maxiter = fill(150, nr)
nreps = fill(150, nr)
maxstep = fill(150, nr)

function meanci(data::Vector{Float64})
    n = length(data)
    m = mean(data)
    s = std(data)
    tstar = 1.962
    hw = tstar * s / sqrt(n)
    m, (m - hw, m + hw)
end

# simple info reward function
ralphas1 = [[1.0, -1/3, -1/3, -1/3],
         [-1/3, 1.0, -1/3, -1/3],
         [-1/3, -1/3, 1.0, -1/3],
         [-1/3, -1/3, -1/3, 1.0]]

# complicated info reward function
vhi = -10.0
vlo = 0.1
ralphas2 = [[1.0, vhi, vhi, vhi],
        [vhi, 1.0, vhi, vhi],
        [vhi, vhi, 1.0, vhi],
        [vhi, vhi, vhi, 1.0],
        [vlo, -vlo/3, -vlo/3, -vlo/3],
        [-vlo/3, vlo, -vlo/3, -vlo/3],
        [-vlo/3, -vlo/3, vlo, -vlo/3],
        [-vlo/3, -vlo/3, -vlo/3, vlo]]

ra = ralphas2
disc = 0.95
uncsize = 0.2
prob_nom = RockIPOMDP(ra, disc)
solver_nom = RPBVISolver(beliefpoints = bs[1],
    max_iterations = maxiter[1])
pol_nom = RPBVI.solve(solver_nom, prob_nom, verbose = true)
bu_nom = updater(pol_nom)
binit_nom = initial_belief_distribution(prob_nom)
prob_r_nom = RockRIPOMDP(ra, disc, uncsize)
pol_r_nom = RPBVI.solve(solver_nom, prob_r_nom, verbose = true)
@show pol_nom.action_map
@show pol_r_nom.action_map
bu_r_nom = updater(pol_r_nom)
binit_r_nom = initial_belief_distribution(prob_r_nom)
simulator = RolloutSimulator(max_steps = maxstep[1])


# #######
# #check belief updater
# b = initialize_belief(bu_r_nom, binit_r_nom)
# a = :check
# o = :bad
# umin, pmin = RPBVI.minutil(prob_r_nom, b.b, a, bu_r_nom.alphas)
# @show pmin[:,:,3]
# bp = update(bu_r_nom, b, a, o)
# @show bp.b
#
# ################################
# # compare robust and nominal belief updates in the worst-case
# blen = 200
# rseed = 8
# bn = initialize_belief(bu_nom, binit_nom)
# alphavecs = pol_r_nom.alphas
# rng = MersenneTwister(rseed)
# srand(8)
# bbhist = Array{Any}(blen)
# bhist = Array{Any}(blen)
# bhist[1] = bn
# bbhist[1] = bn.b
# for i = 2:blen
#     b = bhist[i-1]
#     a = :check
#     s = 20
#     _, o, _ = RPOMDPToolbox.generate_sor_worst(prob_r_nom, b.b, s, a, rng, alphavecs)
#     bp = update(bu_nom, b, a, o)
#     bhist[i] = bp
#     bbhist[i] = bp.b
# end
# hn = [bbhist[i][1] for i=1:blen]
#
# br = initialize_belief(bu_r_nom, binit_r_nom)
# alphavecs = pol_r_nom.alphas
# rng = MersenneTwister(rseed)
# bbhistr = Array{Any}(blen)
# bhistr = Array{Any}(blen)
# bhistr[1] = br
# bbhistr[1] = br.b
# for i = 2:blen
#     b = bhistr[i-1]
#     a = :check
#     s = 20
#     _, o, _ = RPOMDPToolbox.generate_sor_worst(prob_r_nom, b.b, s, a, rng, alphavecs)
#     bp = update(bu_r_nom, b, a, o)
#     bhistr[i] = bp
#     bbhistr[i] = bp.b
# end
# hr = [bbhistr[i][1] for i = 1:blen]
#
# # using Plots; gr()
# plot([hn, hr])

# ####################################
# # check simulators
# simulate(simulator, prob_nom, pol_nom, bu_nom,
#     binit_nom, s_nom) #nn
# simulate(simulator, prob_r_nom, pol_nom, bu_nom,
#     binit_r_nom, s_r_nom) #nr
# simulate(simulator, prob_nom, pol_r_nom, bu_r_nom,
#     binit_nom, s_nom) #rn
# simulate_worst(simulator, prob_r_nom, pol_r_nom, bu_r_nom,
#     binit_r_nom, s_r_nom, pol_r_nom.alphas) #rr

#########################################
# robust vs nom (exp and worst case)
simvals_nn = Array{Float64}(nr, nreps[1],2)
simvals_nr = Array{Float64}(nr, nreps[1],2)
simvals_rn = Array{Float64}(nr, nreps[1],2)
simvals_rr = Array{Float64}(nr, nreps[1],2)
for j = 1:nreps[1]
    print("\rRep $j")
    s_nom = rand(simulator.rng, initial_state_distribution(prob_nom))
    s_r_nom = rand(simulator.rng, initial_state_distribution(prob_r_nom))
    # nominal policy against nominal dynamics
    simvals_nn[1,j,1], simvals_nn[1,j,2] = simulate(simulator,
        prob_nom, pol_nom, bu_nom, binit_nom, s_nom)
    # nominal policy against worst-case dynamics
    simvals_nr[1,j,1], simvals_nr[1,j,2]  = simulate_worst(simulator,
        prob_r_nom, pol_nom, bu_nom, binit_nom, s_r_nom, pol_r_nom.alphas)
    # robust policy against nominal dynamics
    simvals_rn[1,j,1], simvals_rn[1,j,2] = simulate(simulator,
        prob_nom, pol_r_nom, bu_r_nom, binit_r_nom, s_nom)
    # robust policy against worst-case dynamics
    simvals_rr[1,j,1], simvals_rr[1,j,2]  = simulate_worst(simulator,
        prob_r_nom, pol_r_nom, bu_r_nom, binit_r_nom, s_r_nom, pol_r_nom.alphas)
end
value(pol_nom, [0.5, 0.5, 0.0, 0.0])
value(pol_r_nom, [0.5, 0.5, 0.0, 0.0])
m_nn, ci_nn = meanci(simvals_nn[1,:,1])
m_nr, ci_nr = meanci(simvals_nr[1,:,1])
m_rn, ci_rn = meanci(simvals_rn[1,:,1])
m_rr, ci_rr = meanci(simvals_rr[1,:,1])

mp_nn, cip_nn = meanci(simvals_nn[1,:,2])
mp_nr, cip_nr = meanci(simvals_nr[1,:,2])
mp_rn, cip_rn = meanci(simvals_rn[1,:,2])
mp_rr, cip_rr = meanci(simvals_rr[1,:,2])


#################################################
#################################################
# fix for nom exp not matching sim exp
using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
using SimpleProbabilitySets
# using Plots; gr()
const RPBVI = RobustValueIteration

function meanci(data::Vector{Float64})
    n = length(data)
    m = mean(data)
    s = std(data)
    tstar = 1.962
    hw = tstar * s / sqrt(n)
    m, (m - hw, m + hw)
end

# random beliefs
# nr = 1
# nb = 100
# nb2 = 10
# bs1 = [vcat(psample(zeros(2), ones(2)),zeros(2)) for i = 1:nb]
# bs2 = [vcat(zeros(2), psample(zeros(2), ones(2))) for i = 1:nb]
# bs3 = [psample(zeros(4), ones(4)) for i = 1:nb2]
# bs = vcat(bs1, bs2)
# maxiter = 200
# maxstep = 200
# nreps = 100

# random beliefs
nr = 1
bdelt = 0.05
nb2 = 20
bs1 = [vcat([b, 1-b],zeros(2)) for b in 0.0:bdelt:1.0]
bs2 = [vcat(zeros(2), [b, 1-b]) for b in 0.0:bdelt:1.0]
bs3 = [psample(zeros(4), ones(4)) for i = 1:nb2]
bs = vcat(bs1, bs2, bs3)
maxiter = 200
maxstep = 200
nreps = 100

# simple info reward function
ralphas1 = [[1.0, -1/3, -1/3, -1/3],
         [-1/3, 1.0, -1/3, -1/3],
         [-1/3, -1/3, 1.0, -1/3],
         [-1/3, -1/3, -1/3, 1.0]]

# complicated info reward function
vhi = -10.0
vlo = 0.1
ralphas2 = [[1.0, vhi, vhi, vhi],
        [vhi, 1.0, vhi, vhi],
        [vhi, vhi, 1.0, vhi],
        [vhi, vhi, vhi, 1.0],
        [vlo, -vlo/3, -vlo/3, -vlo/3],
        [-vlo/3, vlo, -vlo/3, -vlo/3],
        [-vlo/3, -vlo/3, vlo, -vlo/3],
        [-vlo/3, -vlo/3, -vlo/3, vlo]]

prob_nom = RockIPOMDP(ralphas1, 0.95)
solver_nom = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)
pol_nom = RPBVI.solve(solver_nom, prob_nom)
println(pol_nom.action_map)
bu_nom = updater(pol_nom)
binit_nom = initial_belief_distribution(prob_nom)

srand(11)
nreps = 100
maxstep = 200
simvals_nn = Array{Float64}(nr, nreps, 2)
simulator = RolloutSimulator(max_steps = maxstep)
for j = 1:nreps
    (j % 10 == 0) && print("\rRep $j")
    s_nom = rand(simulator.rng, initial_state_distribution(prob_nom))
    # nominal policy against nominal dynamics
    simvals_nn[1,j,1], simvals_nn[1,j,2] = simulate(simulator,
        prob_nom, pol_nom, bu_nom, binit_nom, s_nom)
end

value(pol_nom, [0.5, 0.5, 0.0, 0.0])
value(pol_nom, [1.0, 0.0, 0.0, 0.0])
value(pol_nom, [0.0, 1.0, 0.0, 0.0])
value(pol_nom, [0.0, 0.0, 0.5, 0.5])
m_nn, ci_nn = meanci(simvals_nn[1,:,1])
mp_nn, cip_nn = meanci(simvals_nn[1,:,2])

xb = 0.0:0.1:1.0
anom = [action(pol_nom, DiscreteBelief(prob_nom, [b, 1-b, 0.0, 0.0])) for b in xb]

using Plots; gr()
valfunc1 = [maximum(dot(pol_nom.alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(pol_nom.alphas))
        for b = 0.0:0.01:1.0]
x = collect(0:0.01:1.0)
vplot = plot(x, valfunc1, xticks = 0:0.1:1, label = "Nominal",
        # title = "Tiger POMDP Value Function",
        xlab = "Belief, P(State = [b, 1-b, 0 0])",
        # xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        legend = :bottomleft,
        # legend = :topright,
        line = :dash,
        linealpha = 0.9)
# plot!(x, valfunc13, color = :red, linealpha = 0.8, label = "Robust: 0.001")
# plot!(x, valfunc21, color = :red, linealpha = 0.4, label = "Robust: 0.4")
# fn = string("value_function_", sname, "_", sversion, ".pdf")
# path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
# savefig(vplot, path)

psim2 = RolloutSimulator(max_steps = 10)
rnum = 28
prob = probs[rnum]
simprob = simprobs[rnum]
sol = policies[rnum]
bu = updater(sol)
binit = initial_belief_distribution(prob)
sinit = rand(psim2.rng, initial_state_distribution(simprob))
simulate_worst(psim2, simprob, sol, bu, binit, sinit, soldynamics[rnum].alphas)

value(policies[26], initial_belief(probs[26]))
value(policies[24], initial_belief(probs[24]))
initial_belief(probs[28]) == initial_belief(probs[26])
policies[26].action_map == policies[28].action_map

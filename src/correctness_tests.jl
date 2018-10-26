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
sr = RPBVISolver(beliefpoints = bs, max_iterations = 1000)
sr2 = RPBVISolver(beliefpoints = bs, max_iterations = 1000)
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
p_tiger = Plots.plot(x, [valsref_tiger, plotvals0_tiger], xticks = 0:0.1:1,
        legend = :top, label = ["SARSOP","PBVI"],
        # title = "Tiger POMDP Value Function",
        xlab = "Belief, P(State = Tiger Left)",
        ylab = "Expected Total Discounted Reward",
        line = (2, [:dash :solid]), alpha = 0.7)
fn = "value_function_tiger.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_tiger, path)

plotvals0_baby = [maximum(dot(polr2.alphas[i], [b, 1-b]) for i = 1:length(polr2.alphas)) for b = 0.0:0.01:1.0]
alphasref_baby = [[-19.6749, -29.6749], [-16.3055, -38.2512]]
valsref_baby = [maximum(dot(alphasref_baby[i], [1-b, b]) for i = 1:length(alphasref_baby))
        for b = 0.0:0.01:1.0]

x = collect(0:0.01:1.0)
p_baby = Plots.plot(x, [valsref_baby, plotvals0_baby], xticks = 0:0.1:1,
        legend = :topright, label = ["SARSOP","PBVI"],
        # title = "Baby POMDP Value Function",
        xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        line = (2,[:dash :solid]), alpha = 0.7)
fn = "value_function_baby.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_baby, path)

# RPBVI Robust POMDPs Tiger
# unc size = 0.001, 0.01, 0.05
srand(8473272)
nr = 5
bs = fill([[b, 1-b] for b in 0.0:0.05:1.0], nr)
ambiguity = [0.1, 0.01, 0.05, 0.1, 0.2]
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

# # plot value
# for i = 1:nr
#     println("Starting loop $i...")
#     prob = TigerRPOMDP(0.95, ambiguity[i])
#     sr = RPBVISolver(beliefpoints = bs[i], max_iterations = maxiter[i])
#     println("Calculating policy...")
#     polr = RobustValueIteration.solve(sr, prob)
#     plotvals_tiger[i] = [maximum(dot(polr.alphas[j], [1-b, b]) for j = 1:length(polr.alphas))
#         for b = 0.0:0.01:1.0]
# end

function meanci(data::Vector{Float64})
    n = length(data)
    m = mean(data)
    s = std(data)
    tstar = 1.962
    hw = tstar * s / sqrt(n)
    m, (m - hw, m + hw)
end

nreps[1] = 500
uncsize = 0.05
prob_nom = TigerPOMDP(0.95)
solver_nom = RPBVISolver(beliefpoints = bs[1], max_iterations = maxiter[1])
pol_nom = RPBVI.solve(solver_nom, prob_nom)
bu_nom = updater(pol_nom)
prob_r_nom = SimpleTigerRPOMDP(0.95, uncsize)
pol_r_nom = RPBVI.solve(solver_nom, prob_r_nom)
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
value(pol_nom, [0.5, 0.5])
value(pol_r_nom, [0.5, 0.5])
m_nn, ci_nn = meanci(simvals_nn[1,:,1])
m_nr, ci_nr = meanci(simvals_nr[1,:,1])
m_rn, ci_rn = meanci(simvals_rn[1,:,1])
m_rr, ci_rr = meanci(simvals_rr[1,:,1])


# sim value
srand(8473272)
for i = 1:1
    println("Loop $i")
    prob = TigerRPOMDP(0.95, ambiguity[i])
    sr = RPBVISolver(beliefpoints = bs[i], max_iterations = maxiter[i])
    println("Calculating policy...")
    polr = RobustValueIteration.solve(sr, prob)
    bur = updater(polr)
    policyvalue_tiger[i] = value(polr, [0.5,0.5])
    println("Simulating value...")
    simvals_tiger_temp = Vector{}(nreps[i])
    simps_tiger_temp = Vector{}(nreps[i])
    simvals_tiger_nom_temp = Vector{}(nreps[i])
    simps_tiger_nom_temp = Vector{}(nreps[i])
    simulator = RolloutSimulator(max_steps = maxstep[i])
    for j = 1:nreps[i]
        (j % 10 == 0) && print("\rRep $j")
        simvals_tiger_temp[j], simps_tiger_temp[j]  = simulate_worst(simulator,
            prob, polr, bur, polr.alphas)
        simvals_tiger_nom_temp[j], simps_tiger_nom_temp[j]  = simulate_worst(simulator,
            prob, pol_nom, bu_nom, polr.alphas)
    end
    simvals_tiger[i] = simvals_tiger_temp
    simps_tiger[i] = simps_tiger_temp
    simvals_tiger_nom[i] = simvals_tiger_nom_temp
    simps_tiger_nom[i] = simps_tiger_nom_temp
    m_tiger[i] = mean(simvals_tiger[i])
    m_nom_tiger[i] = mean(simvals_tiger_nom[i])
    s_tiger[i] = std(simvals_tiger[i])
    tstar = 1.962
    ci_tiger[i] = (m_tiger[i] - tstar * s_tiger[i] / sqrt(nreps[i]), m_tiger[i] + tstar * s_tiger[i] / sqrt(nreps[i]))
    ci_nom_tiger[i] = (m_nom_tiger[i] - tstar * std(simvals_tiger_nom[i]) / sqrt(nreps[i]),
        m_nom_tiger[i] + tstar * std(simvals_tiger_nom[i]) / sqrt(nreps[i]))
end

# reference values
alphasref_tiger = [[-81.5975, 28.4025],
        [3.01448, 24.6954],
        [24.6954, 3.01452],
        [28.4025, -81.5975],
        [19.3711, 19.3711]]
valsref_tiger = [maximum(dot(alphasref_tiger[i], [b, 1-b]) for i = 1:length(alphasref_tiger))
        for b = 0.0:0.01:1.0]

x = collect(0:0.01:1.0)
p_tiger_robust = plot(x, valsref_tiger, xticks = 0:0.1:1, label = "SARSOP",
        # title = "Tiger POMDP Value Function",
        xlab = "Belief, P(State = Tiger Left)",
        ylab = "Expected Total Discounted Reward",
        legend = :bottomleft,
        line = :dash,
        linealpha = 0.9)
plot!(x, plotvals0_tiger, color = :red, linealpha = 0.8, label = "PBVI")
plot!(x, plotvals_tiger[1], color = :red, linealpha = 0.7, label = "RPBVI: 0.001")
plot!(x, plotvals_tiger[2], color = :red, linealpha = 0.6, label = "RPBVI: 0.01")
plot!(x, plotvals_tiger[3], color = :red, linealpha = 0.5, label = "RPBVI: 0.05")
plot!(x, plotvals_tiger[4], color = :red, linealpha = 0.4, label = "RPBVI: 0.1")
plot!(x, plotvals_tiger[5], color = :red, linealpha = 0.3, label = "RPBVI: 0.2")
fn = "value_function_tiger_robust_2.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_tiger_robust, path)

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
        line = :dash,
        linealpha = 1.0)
# plot!(x, plotvals0_baby, color = :red, linealpha = 0.8, label = "PBVI")
plot!(x, plotvals_baby[1], color = :red, linealpha = 0.7, label = "RPBVI: 0.001")
plot!(x, plotvals_baby[2], color = :red, linealpha = 0.6, label = "RPBVI: 0.01")
plot!(x, plotvals_baby[3], color = :red, linealpha = 0.5, label = "RPBVI: 0.05")
plot!(x, plotvals_baby[4], color = :red, linealpha = 0.4, label = "RPBVI: 0.1")
plot!(x, plotvals_baby[5], color = :red, linealpha = 0.3, label = "RPBVI: 0.2")
fn = "value_function_baby_robust.pdf"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_baby_robust, path)


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
ambiguity = [0.1, 0.01, 0.05, 0.1, 0.2]
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

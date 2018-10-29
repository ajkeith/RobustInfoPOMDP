using RPOMDPModels, RPOMDPs, RPOMDPToolbox
using RobustValueIteration
using SimpleProbabilitySets
using DataFrames, ProgressMeter, CSV, BenchmarkTools
const RPBVI = RobustValueIteration
TOL = 1e-6

# intialize problems
p = CyberPOMDP()
ip = CyberIPOMDP()
rp = CyberRPOMDP()
rip = CyberRIPOMDP()
tip = CyberTestIPOMDP()

# select belief
srand(7971023)
s0 = fill(1/27, 27);
s1 = zeros(27); s1[14] = 1;
s2 = zeros(27); s2[1] = 1;
s32 = zeros(27); s32[1] = 0.5; s32[2] = 0.5;
s42 = zeros(27); s42[1] = 0.5; s42[4] = 0.5;
s52 = zeros(27); s52[1] = 0.5; s52[10] = 0.5;
s33 = zeros(27); s33[1] = 0.5; s33[3] = 0.5;
s43 = zeros(27); s43[1] = 0.5; s43[7] = 0.5;
s53 = zeros(27); s53[1] = 0.5; s53[19] = 0.5;
s6 = zeros(27); s6[1] = 0.5; s6[6] = 0.5;
s7 = zeros(27); s7[1] = 0.5; s7[22] = 0.5;
ss = [s0, s1, s2, s32, s42, s52, s33, s43, s53, s6, s7]
nS = length(states(p))
nrand = 5
bs = Vector{Vector{Float64}}(length(ss) + nrand)
for i = 1:length(ss)
    bs[i] = ss[i]
end
for i = (length(ss)+1):(length(ss) + nrand)
  bs[i] = psample(zeros(nS), ones(nS))
end
push!(bs, vcat(fill(0.0, nS - 1), 1.0))
push!(bs, fill(1/nS, nS))

# intialize solver
solver = RPBVISolver(beliefpoints = bs, max_iterations = 20)

# solve
srand(5917293)
@time solip = RPBVI.solve(solver, ip);
@time solrip = RPBVI.solve(solver, rip);

# calculate values
e1 = zeros(27); e1[1] = 1
println("Standard Value: ", policyvalue(solip, e1))
println("Robust Value: ", policyvalue(solrip, e1))

# ipomdp and ripomdp actions for some interesting states
actionind = ["unif", "2,2,2", "1,1,1", "1,1,1 - 1,1,2", "1,1,1 - 1,2,1", "1,1,1 - 2,1,1",
        "1,1,1 - 1,1,3" ,"1,1,1 - 1,3,1", "1,1,1 - 3,1,1",
        "1,1,1 - 1,2,3", "1,1,1 - 3,2,1"]
dbsip = [DiscreteBelief(ip, states(ip), s) for s in ss]
asip = [action(solip, db) for db in dbsip]
dbsrip = [DiscreteBelief(rip, states(rip), s) for s in ss]
asrip = [action(solrip, db) for db in dbsrip]
dbstip = [DiscreteBelief(tip, states(tip), s) for s in ss]
astip = [action(soltip, db) for db in dbstip]
actiondata = DataFrame(Belief = actionind, StdAction = asip,
                            RobustAction = asrip, OffNominalAction = astip)
@show actiondata

# sim values for nominal and robust solutions for off-nominal case
ntest = 2
nreps = 40
nsteps = 20
psim = RolloutSimulator(max_steps = nsteps)
simvals = [Vector{Float64}(nreps) for _ in 1:ntest] # simulated values
simps = [Vector{Float64}(nreps) for _ in 1:ntest] # simulated percent correct
ves = Array{Float64}(ntest) # expected values
vms = Array{Float64}(ntest) # mean of sim values
vss = Array{Float64}(ntest) # std dev of sim values
vmins = Array{Float64}(ntest) # min of sim values
pms = Array{Float64}(ntest) # mean percent correct
pss = Array{Float64}(ntest) # std percent correct

rseed = 92378432
simprob = rip
simdynamics = :worst
for i in 1:1
    # println("Run ", ceil(Int, i / 2))
    buip, burip = updater(solip), updater(solrip)
    println("Nominal")
    srand(rseed)
    binitip = initial_belief_distribution(ip)
    sinitrip = rand(psim.rng, initial_state_distribution(simprob))
    @showprogress 1 "Simulating nominal value..." for j = 1:nreps
        if simdynamics == :worst
            sv, sp = simulate_worst(psim, simprob, solip, buip, binitip, sinitrip, solrip.alphas)
        else
            sv, sp = simulate(psim, simprob, solip, buip)
        end
        simvals[i][j] = sv
        simps[i][j] = sp
    end
    vms[i] = mean(simvals[i])
    vss[i] = std(simvals[i])
    vmins[i] = minimum(simvals[i])
    pms[i] = mean(simps[i])
    pss[i] = std(simps[i])
    println("Robust")
    srand(rseed)
    binitrip = initial_belief_distribution(rip)
    @showprogress 1 "Simulating robust value..." for j = 1:nreps
        sv, sp = simulate(psim, simprob, solrip, buip)
        if simdynamics == :worst
            sv, sp = simulate_worst(psim, simprob, solrip, burip, binitrip, sinitrip, solrip.alphas)
        else
            sv, sp = simulate(psim, simprob, solrip, buip)
        end
        simvals[i+1][j] = sv
        simps[i+1][j] = sp
    end
    vms[i+1] = mean(simvals[i+1])
    vss[i+1] = std(simvals[i+1])
    vmins[i+1] = minimum(simvals[i+1])
    pms[i+1] = mean(simps[i+1])
    pss[i+1] = std(simps[i+1])
end

sname = "assessment"
sversion = "4.0"
ves = [policyvalue(solip, e1), policyvalue(solrip, e1)]
rdata = DataFrame(ID = ["Nominal", "Robust"], ExpValue = ves,
            SimMean = vms, SimStd = vss, SimMin = vmins,
            SimPercentMean = pms, SimPercentStd = pss)
simdata = DataFrame(NominalSim = simvals[1], RobustSim = simvals[2])
@show rdata


path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnactions = string("exp_actions_", sname, "_", sversion, ".csv")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnactions), actiondata)
CSV.write(joinpath(path, fnresults), rdata)
CSV.write(joinpath(path, fnsim), simdata)


######################################################
# Results
######################################################

# Version 2.0 - Complete
# sol iter = 10
# sim steps = 10
# sim reps = 5
# problem type: robust
# problem dynamics: worst case
# solution policy: nominal, robust, off-nominal (respectively)

# Version 3.0 - Complete
# sol iter = 100
# sim steps = 20
# sim reps = 40
# problem type: robust
# problem dynamics: worst case
# solution policy: nominal, robust, off-nominal (respectively)

# Version 3.1 - Complete
# sol iter = 17
# sim steps = 17
# sim reps = 15
# problem type: robust
# problem dynamics: worst case
# solution policy: nominal, robust, off-nominal (respectively)
# belief points: selected + nrand = 20

# Versions 2.0 to 3.1 all use a bad generate_sor_worst...
# need to fix generate_sor_worst and redo all of them


######################################################
# Plots
######################################################

# # how to make each line a different sytle?
# using StatPlots
# @df simdata density([:NominalSim, :RobustSim, :OffNominal],
#     labels = ["Standard Policy (Nominal)","Robust Policy","Standard Policy (Off-Nominal)"],
#     xlim = [1.5, 2.5],
#     style = [:solid, :solid, :dot],
#     title = "Worst-Case Belief-Reward (n = 40)", xlab = "Empirical Total Discounted Reward", ylab = "Density")


######################################################
# Checks
######################################################

# generate_sor_worst investigation

# rip = CyberRIPOMDP()
# simprob = rip
# srand(7971023)
# s0 = fill(1/27, 27);
# s1 = zeros(27); s1[14] = 1;
# s2 = zeros(27); s2[1] = 1;
# s32 = zeros(27); s32[1] = 0.5; s32[2] = 0.5;
# s42 = zeros(27); s42[1] = 0.5; s42[4] = 0.5;
# s52 = zeros(27); s52[1] = 0.5; s52[10] = 0.5;
# s33 = zeros(27); s33[1] = 0.5; s33[3] = 0.5;
# s43 = zeros(27); s43[1] = 0.5; s43[7] = 0.5;
# s53 = zeros(27); s53[1] = 0.5; s53[19] = 0.5;
# s6 = zeros(27); s6[1] = 0.5; s6[6] = 0.5;
# s7 = zeros(27); s7[1] = 0.5; s7[22] = 0.5;
# ss = [s0, s1, s2, s32, s42, s52, s33, s43, s53, s6, s7]
# nS = length(states(rip))
# nrand = 1
# bs = Vector{Vector{Float64}}(length(ss) + nrand)
# for i = 1:length(ss)
#     bs[i] = ss[i]
# end
# for i = (length(ss)+1):(length(ss) + nrand)
#   bs[i] = psample(zeros(nS), ones(nS))
# end
# push!(bs, vcat(fill(0.0, nS - 1), 1.0))
# push!(bs, fill(1/nS, nS))
#
# # intialize solver
# solver = RPBVISolver(beliefpoints = bs, max_iterations = 10)
# solrip = RobustValueIteration.solve(solver, rip)
# burip = updater(solrip)
# psim = RolloutSimulator(max_steps = 5)
# rng = MersenneTwister(0)
# b = s0
# s = [2,2,2]
# a = [1,1]
# RPOMDPToolbox.generate_sor_worst(rip, b, s, a, rng, solrip.alphas)[1]
# simulate_worst(psim, simprob, solrip, burip, solrip.alphas)
#

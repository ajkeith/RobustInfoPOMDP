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

# select belief points
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
bs = Vector{Vector{Float64}}(nS)
for i = 1:length(ss)
    bs[i] = ss[i]
end
for i = (length(ss)+1):(nS-1)
  bs[i] = psample(zeros(nS), ones(nS))
end
bs[nS] = vcat(fill(0.0, nS - 1), 1.0)
push!(bs, fill(1/nS, nS))

# intialize solver
solver = RPBVISolver(beliefpoints = bs, max_iterations = 10)

# solve
@time solip = RPBVI.solve(solver, ip); # 352 sec at 100 iter
@time solrip = RPBVI.solve(solver, rip); # 1339 sec at 10 iter
@time soltip = RPBVI.solve(solver, tip)

# calculate values
e1 = zeros(27); e1[1] = 1
println("Standard Value: ", policyvalue(solip, e1))
println("Robust Value: ", policyvalue(solrip, e1))
println("Off Nominal Precise Value: ", policyvalue(soltip, e1))

# ipomdp and ripomdp actions for some interesting states
actionind = ["unif", "2,2,2", "1,1,1", "1,1,1 - 1,1,2", "1,1,1 - 1,2,1", "1,1,1 - 2,1,1",
        "1,1,1 - 1,1,3" ,"1,1,1 - 1,3,1", "1,1,1 - 3,1,1",
        "1,1,1 - 1,2,3", "1,1,1 - 3,2,1" ]
dbsip = [DiscreteBelief(ip, states(ip), s) for s in ss]
asip = [action(solip, db) for db in dbsip]
dbsrip = [DiscreteBelief(rip, states(rip), s) for s in ss]
asrip = [action(solrip, db) for db in dbsrip]
dbstip = [DiscreteBelief(tip, states(tip), s) for s in ss]
astip = [action(soltip, db) for db in dbstip]
actiondata = DataFrame(Belief = actionind, StdAction = asip, RobustAction = asrip, OffNominalAction = astip)
@show actiondata

# sim values for nominal and robust solutions for off-nominal case
ipoff = CyberTestIPOMDP()
ntest = 3
nreps = 5
nsteps = 10
psim = RolloutSimulator(max_steps = nsteps)
simvals = Array{Array{Float64}}(ntest) # simulated values
ves = Array{Float64}(ntest) # expected values
vms = Array{Float64}(ntest) # mean of sim values
vss = Array{Float64}(ntest) # std dev of sim values
vmins = Array{Float64}(ntest) # min of sim values

srand(2394872)
for i in 1:1
    # println("Run ", ceil(Int, i / 2))
    buip, burip = updater(solip), updater(solrip)
    simvals[i] = @showprogress 1 "Simulating nominal value..." [simulate(psim, ipoff, solip, buip) for j=1:nreps]
    vms[i] = mean(simvals[i])
    vss[i] = std(simvals[i])
    vmins[i] = minimum(simvals[i])
    # simvals[i+1] = @showprogress 1 "Simulating robust value..." [simulate(psim, ipoff, solrip, burip) for j=1:nreps]
    # vms[i+1] = mean(simvals[i+1])
    # vss[i+1] = std(simvals[i+1])
    # vmins[i+1] = minimum(simvals[i+1])
    simvals[i+2] = @showprogress 1 "Simulating offnominal value..." [simulate(psim, ipoff, soltip, burip) for j=1:nreps]
    vms[i+2] = mean(simvals[i+2])
    vss[i+2] = std(simvals[i+2])
    vmins[i+2] = minimum(simvals[i+2])
end

sname = "assessment"
sversion = "1.2"
ves = [policyvalue(solip, e1), policyvalue(solrip, e1), policyvalue(soltip, e1)]
rdata = DataFrame(ID = ["Nominal", "Robust","OffNominal"], ExpectedValue = ves, SimMean = vms, SimStd = vss, SimMin = vmins)
simdata = DataFrame(NominalSim = simvals[1], RobustSim = simvals[2], OffNominal = simvals[3])
@show rdata
@show simdata


path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
fnactions = string("exp_actions_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnresults), rdata)
CSV.write(joinpath(path, fnsim), simdata)
CSV.write(joinpath(path, fnactions), actiondata)



# ########################################
# #
# # Automated Data Collection
# #
# ########################################
#
# # load packages
# using RPOMDPs, RPOMDPModels, RPOMDPToolbox, RobustValueIteration
# using DataFrames, Query, CSV, ProgressMeter
# const rpbvi = RobustValueIteration
#
# # setup results structures
# # solutions: holds policies and full sim results
# # data: holds solution values and copmutation time
# factors = DataFrame(ID = Int[], Problem = String[], Short_Name = String[], Solution = String[], Uncertainty_Size = Float64[], Dynamics = String[])
# probnames = ["Assessment"]
# shortnames = ["assessment"]
# soltypes = ["Standard", "Robust"]
# uncsizes = [0.025, 0.1, 0.3]
# dyntypes = ["Nominal", "Ambiguous"]
# respsols = ["Policy", "Simulation Values"]
# respdata = ["Solution Value", "Simulation Value (Mean)", "Simulation Value (Std Dev)", "Computation Time"]
# headerfacts = ["ID","Problem", "Short Name", "Solution", "Uncertainty Size", "Dynamics"]
# headersols = vcat(factors, respsols)
# headerdata = vcat(factors, respdata)
#
# ind = 0
# for pname in probnames, sol in soltypes, u in uncsizes, d in dyntypes
#     ind += 1
#     sname = shortnames[findin(probnames,[pname])[1]]
#     push!(factors, [ind, pname, sname, sol, u, d])
# end
#
# for i in 1:size(factors,1)
#     if (factors[:Solution][i] == "Standard") && (factors[:Dynamics][i] == "Nominal")
#         factors[:Uncertainty_Size][i] = 0.0
#     end
# end
# new_factors = unique(factors[:, 2:6])
#
# # set up problems - WIP
# function build(sname, robust, err)
#     prob = nothing
#     if sname == "assessment"
#         prob = (robust == "Robust") ? CyberRIPOMDP(err) : CyberIPOMDP()
#     end
#     prob
# end
#
# sname = "assessment"
# dfexp = @from run in new_factors begin
#             @where run.Short_Name == sname
#             @select run
#             @collect DataFrame
# end
#
# nrows = size(dfexp,1)
# probs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(nrows)
# simprobs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(nrows)
# for i = 1:nrows
#     sname = dfexp[:Short_Name][i]
#     robust = dfexp[:Solution][i]
#     err = dfexp[:Uncertainty_Size][i]
#     probs[i] = build(sname, robust, err)
#     simrobust = (dfexp[:Dynamics][i] == "Nominal") ? "Standard" : "Robust"
#     simprobs[i] = build(sname, simrobust, err)
# end
#
# # compare cyber ip to cyber rip
# nS = 27
# bs = Vector{Vector{Float64}}(nS)
# bs[1] = vcat(1.0, fill(0.0, nS - 1))
# bs[nS] = vcat(fill(0.0, nS - 1), 1.0)
# for i = 2:(nS-1)
#   # bs[i] = vcat(fill(0.0, i - 1), 1.0, fill(0.0, nS - i))
#   bs[i] = psample(zeros(nS), ones(nS))
# end
# push!(bs, fill(1/nS, nS))
# solver = RPBVISolver(beliefpoints = bs[19:28], max_iterations = 100)
# solip = RPBVI.solve(solver, ip)
# solrip = RPBVI.solve(solver, rip)
#
#
# # ntest = size(dfexp,1)
# ntest = size(dfexp,1)
# nreps = 3
# nsteps = 10
# psim = RolloutSimulator(max_steps = nsteps)
# policies = Array{AlphaVectorPolicy}(ntest) # solution policies
# simvals = Array{Array{Float64}}(ntest) # simulated values
# ves = Array{Float64}(ntest) # expected values
# vms = Array{Float64}(ntest) # mean of sim values
# vss = Array{Float64}(ntest) # std dev of sim values
#
# # rollout generates state/obs/rewards based on the sim problem, and beliefs based on the solution policy
# srand(923475)
# for i in 1:ntest
#     println("Run $i")
#     println("Solution Type: ", new_factors[:Solution][i], " Dynamics")
#     prob = probs[i]
#     simprob = simprobs[i]
#     sol = rpbvi.solve(solver, prob)
#     policies[i] = sol
#     ves[i] = value(sol, initial_belief(prob))
#     bu = updater(sol)
#     println("Simulation Type: ", new_factors[:Dynamics][i], " ", typeof(simprob))
#     simvals[i] = @showprogress 1 "Simulating value..." [simulate(psim, simprob, sol, bu) for j=1:nreps]
#     vms[i] = mean(simvals[i])
#     vss[i] = std(simvals[i])
# end
#
# ids = collect(1:size(dfexp,1))
# dfexp[:ID] = ids
# rdata = DataFrame(ID = ids, ExpectedValue = ves, SimMean = vms, SimStd = vss)
# df = join(dfexp, rdata, on = :ID)
# simdata = hcat(ids, hcat(simvals...)') |> DataFrame
# path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
# fnresults = string("exp_results_", sname, ".csv")
# fnsim = string("exp_sim_values_", sname, ".csv")
# CSV.write(joinpath(path, fnresults), df)
# CSV.write(joinpath(path, fnsim), simdata)

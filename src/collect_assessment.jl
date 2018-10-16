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
bs = Vector{Vector{Float64}}(length(ss)+5)
for i = 1:length(ss)
    bs[i] = ss[i]
end
for i = (length(ss)+1):(length(ss)+5)
  bs[i] = psample(zeros(nS), ones(nS))
end
push!(bs, vcat(fill(0.0, nS - 1), 1.0))
push!(bs, fill(1/nS, nS))

# intialize solver
solver = RPBVISolver(beliefpoints = bs, max_iterations = 100)

# solve
srand(5917293)
@time solip = RPBVI.solve(solver, ip); # 10 sec at 10 iter
@time solrip = RPBVI.solve(solver, rip); # 718 sec at 10 iter
@time soltip = RPBVI.solve(solver, tip); # 10 sec at 10 iter

# calculate values
e1 = zeros(27); e1[1] = 1
println("Standard Value: ", policyvalue(solip, e1))
println("Robust Value: ", policyvalue(solrip, e1))
println("Off Nominal Precise Value: ", policyvalue(soltip, e1))

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
ntest = 3
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
    buip, burip, butip = updater(solip), updater(solrip), updater(soltip)
    println("Nominal")
    srand(rseed)
    @showprogress 1 "Simulating nominal value..." for j = 1:nreps
        if simdynamics == :worst
            sv, sp = simulate_worst(psim, simprob, solip, buip, solrip.alphas)
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
    @showprogress 1 "Simulating robust value..." for j = 1:nreps
        sv, sp = simulate(psim, simprob, solrip, buip)
        if simdynamics == :worst
            sv, sp = simulate_worst(psim, simprob, solrip, burip, solrip.alphas)
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
    println("OffNominal")
    srand(rseed)
    @showprogress 1 "Simulating off-nominal value..." for j = 1:nreps
        if simdynamics == :worst
            sv, sp = simulate_worst(psim, simprob, soltip, butip, solrip.alphas)
        else
            sv, sp = simulate(psim, simprob, soltip, butip)
        end
        simvals[i+2][j] = sv
        simps[i+2][j] = sp
    end
    vms[i+2] = mean(simvals[i+2])
    vss[i+2] = std(simvals[i+2])
    vmins[i+2] = minimum(simvals[i+2])
    pms[i+2] = mean(simps[i+2])
    pss[i+2] = std(simps[i+2])
end

sname = "assessment"
sversion = "3.0"
ves = [policyvalue(solip, e1), policyvalue(solrip, e1),
        policyvalue(soltip, e1)]
rdata = DataFrame(ID = ["Nominal", "Robust","OffNominal"], ExpValue = ves,
            SimMean = vms, SimStd = vss, SimMin = vmins,
            SimPercentMean = pms, SimPercentStd = pss)
simdata = DataFrame(NominalSim = simvals[1], RobustSim = simvals[2],
            OffNominal = simvals[1])
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

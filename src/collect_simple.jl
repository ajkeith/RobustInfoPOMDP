using RPOMDPModels, RPOMDPs, RPOMDPToolbox
using RobustValueIteration
using SimpleProbabilitySets
using DataFrames, ProgressMeter, CSV, BenchmarkTools
const RPBVI = RobustValueIteration
TOL = 1e-6

# intialize problems
ip = SimpleIPOMDP(0.8, 0.7, 0.66, 0.85)
rip = SimpleRIPOMDP(0.8, 0.7, 0.66, 0.85, 0.15)
tip = SimpleIPOMDP(0.65, 0.85, 0.51, 0.7)

# select belief points
ss = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
        [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0]]
bs = Vector{Vector{Float64}}(length(ss))
for i = 1:length(ss)
    bs[i] = ss[i]
end

# intialize solver
solver = RPBVISolver(beliefpoints = bs, max_iterations = 500)

# solve
@time solip = RPBVI.solve(solver, ip);
@time solrip = RPBVI.solve(solver, rip);
@time soltip = RPBVI.solve(solver, tip);

# calculate values
nS = length(states(ip))
e1 = zeros(nS); e1[1] = 1
println("Standard Value: ", policyvalue(solip, e1))
println("Robust Value: ", policyvalue(solrip, e1))
println("Off Nominal Precise Value: ", policyvalue(soltip, e1))

# ipomdp and ripomdp actions for some interesting states
actionind = ss
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
nreps = 50
nsteps = 100
# IMPORTANT: need to find a way to have both default and worst sims
# right now I can only make worst sims (but some important and good results need the default)
# psim_default = RolloutSimulator(max_steps = nsteps)
psim_worst = RolloutSimulator(solrip.alphas, max_steps = nsteps)
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
psim = psim_worst
for i in 1:1
    # println("Run ", ceil(Int, i / 2))
    println("Nominal")
    srand(rseed)
    buip, burip, butip = updater(solip), updater(solrip), updater(soltip)
    @showprogress 1 "Simulating nominal value..." for j = 1:nreps
        sv, sp = simulate(psim, simprob, solip, burip)
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
        sv, sp = simulate(psim, simprob, soltip, butip)
        simvals[i+2][j] = sv
        simps[i+2][j] = sp
    end
    vms[i+2] = mean(simvals[i+2])
    vss[i+2] = std(simvals[i+2])
    vmins[i+2] = minimum(simvals[i+2])
    pms[i+2] = mean(simps[i+2])
    pss[i+2] = std(simps[i+2])
end

sname = "simple"
sversion = "2.1"
ves = [policyvalue(solip, e1), policyvalue(solrip, e1),
        policyvalue(soltip, e1)]
rdata = DataFrame(ID = ["Nominal", "Robust","OffNominal"], ExpValue = ves,
                SimMean = vms, SimStd = vss, SimMin = vmins,
                SimPercentMean = pms, SimPercentStd = pss)
simdata = DataFrame(NominalSim = simvals[1], RobustSim = simvals[2],
                        OffNominal = simvals[1])
@show rdata



path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
fnactions = string("exp_actions_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnresults), rdata)
CSV.write(joinpath(path, fnsim), simdata)
CSV.write(joinpath(path, fnactions), actiondata)


########### testing ############
rng = MersenneTwister(0)
b = [0.6, 0.4]

dbip = DiscreteBelief(ip, states(ip), b)
buip = updater(solip)
dbtip = DiscreteBelief(tip, states(tip), b)
butip = updater(soltip)
dbrip = DiscreteBelief(rip, states(rip), b)
burip = updater(solrip)
s = :left
a = :both
o = :LR


dbip = initialize_belief(buip, initial_state_distribution(ip))
dbrip = initialize_belief(burip, initial_state_distribution(rip))

dbip = DiscreteBelief(ip, states(ip), [0.745, 0.255])
dbrip = DiscreteBelief(rp, states(rip), [0.745, 0.255])
a = :single
o = :LL

update(buip, dbip, a, o).b
update(butip, dbtip, a, o).b

generate_sor(rip, dbip.b, s, a, rng)

sp = :left
observation(rip, a, sp)
psample(observation(rip, a, sp)...)

nsteps = 10
psim = RolloutSimulator(solrip.alphas, max_steps = nsteps)
simulate(psim, rip, solrip, buip)

generate_sor_worst(rip, dbip.b, s, a, rng, solrip.alphas)

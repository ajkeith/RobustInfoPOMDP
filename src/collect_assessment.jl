using RPOMDPModels, RPOMDPs, RPOMDPToolbox
using RobustValueIteration
using SimpleProbabilitySets
using DataFrames, ProgressMeter, CSV, BenchmarkTools, StatsBase
const RPBVI = RobustValueIteration
TOL = 1e-6

# set experiment parameters
sname = "assessment"
sversion = "7.1"
disc = 0.9
maxiter = 45
nbrand = 40
nreps = 10
nsteps = 45
nrows = 3

# intialize problems
ip = CyberIPOMDP(disc)
rip = CyberRIPOMDP(disc)
tip = CyberTestIPOMDP(disc)

function meanci(data::Vector{Float64})
    n = length(data)
    m = mean(data)
    s = std(data)
    tstar = 1.962
    hw = tstar * s / sqrt(n)
    (m - hw, m + hw)
end

# # select belief
# srand(7971023)
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
# ss = [s1, s2, s32, s42, s52, s33, s43, s53, s6, s7]
# nS = length(states(ip))
# bs = Vector{Vector{Float64}}(length(ss) + nbrand)
# for i = 1:length(ss)
#     bs[i] = ss[i]
# end
# for i = (length(ss)+1):(length(ss) + nbrand)
#     @show i
#     bs[i] = psample(zeros(nS), ones(nS))
# end
# push!(bs, vcat(fill(0.0, nS - 1), 1.0))
# push!(bs, fill(1/nS, nS))
# bsmatrix = Array{Float64}(length(bs), length(bs[1]))
# for i = 1:length(bs)
#     for j = 1:length(bs[i])
#         bsmatrix[i,j] = bs[i][j]
#     end
# end
# pathbp = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
# fnbp = string("belief_points_", sname, "_", sversion, ".csv")
# CSV.write(joinpath(pathbp, fnbp), DataFrame(bsmatrix))

# load bs (if previous bs was large)
pathbp = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnbp = string("belief_points_", sname, "_7.0.csv")
bsdf = CSV.read(joinpath(pathbp, fnbp))
bsmatrix = convert(Array, bsdf[:,:])
numrowbs, numcolbs = size(bsmatrix)
bs = Vector{Vector{Float64}}(numrowbs)
for i = 1:numrowbs
    bs[i] = bsmatrix[i,:]
end
length(bs) == nbrand + 12

# intialize solver
solver = RPBVISolver(beliefpoints = bs, max_iterations = maxiter)

# solve
srand(5917293)
@time solip = RPBVI.solve(solver, ip, verbose = true);
@time solrip = RPBVI.solve(solver, rip, verbose = true);
@time soltip = RPBVI.solve(solver, tip, verbose = true);

# calculate values
e1 = zeros(27); e1[1] = 1
println("Standard Value: ", policyvalue(solip, e1))
println("Robust Value: ", policyvalue(solrip, e1))
println("Off Nominal Precise Value: ", policyvalue(soltip, e1))

# ipomdp and ripomdp actions for some interesting states
actionind = ["2,2,2", "1,1,1", "1,1,1 - 1,1,2", "1,1,1 - 1,2,1", "1,1,1 - 2,1,1",
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
# actiondata = DataFrame(Belief = actionind, StdAction = asip, OffNominalAction = astip)
@show actiondata

# # save policies
# policyname = "off_nominal"
# bpversion = "7.0"
# policyversion = sversion
# sol = soltip
# prob_policy = tip
# alphamatrix = Array{Float64}(length(bs), length(bs[1]))
# for i = 1:length(bs)
#     for j = 1:length(bs[i])
#         alphamatrix[i,j] = sol.alphas[i][j]
#     end
# end
# actionmatrix = action_index.(prob_policy, sol.action_map)
# pathbp = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
# fnalphas = string("policy_alphas_", policyname, "_bp", bpversion, "_results", sversion, ".csv")
# fnactions = string("policy_actions_", policyname, "_bp", bpversion, "_results", sversion, ".csv")
# CSV.write(joinpath(pathbp, fnalphas), DataFrame(alphamatrix))
# CSV.write(joinpath(pathbp, fnactions), DataFrame(action_ind = actionmatrix))

# set sim structures
nreps = 25
nsteps = 100
bus = [updater(solip), updater(solrip), updater(soltip)]
binits = [SparseCat(states(ip), initial_belief(ip)),
    SparseCat(states(rip), initial_belief(rip)),
    SparseCat(states(tip), initial_belief(tip))]
sols = [solip, solrip, soltip]
psim = RolloutSimulator(max_steps = nsteps)
simvals = [Vector{Float64}(nreps) for _ in 1:nrows] # simulated values
simps = [Vector{Float64}(nreps) for _ in 1:nrows] # simulated percent correct
ves = Array{Float64}(nrows) # expected values
vms = Array{Float64}(nrows) # mean of sim values
vss = Array{Float64}(nrows) # std dev of sim values
vci = Array{Tuple{Float64,Float64}}(nrows) # 95% ci of mean of sim values
pms = Array{Float64}(nrows) # mean percent correct
pss = Array{Float64}(nrows) # std percent correct
pci = Array{Tuple{Float64,Float64}}(nrows) # 95% ci of mean of sim percent correct

# run simluation
simprob = tip
if simprob == tip
    simprobnames = fill("Off Nominal", nrows)
elseif simprob == rip
    simprobnames = fill("Robust", nrows)
elseif simprob == ip
    simprobnames = fill("Nominal", nrows)
else
    println("Error: Unexpected simprob")
    simprobnanmes = fill("Undefined", nrows)
end
simdynamics = (simprob == rip) ? :worst : :precise
rseed = 92378432
sinit = rand(psim.rng, initial_state_distribution(simprob))
for i in 1:nrows
    srand(rseed)
    println("Run $i...")
    @showprogress 1 "Simulating value..." for j = 1:nreps
        if simdynamics == :worst
            sv, sp = simulate_worst(psim, simprob, sols[i], bus[i], binits[i], sinit, solrip.alphas)
        else
            sv, sp = simulate(psim, simprob, sols[i], bus[i], binits[i], sinit)
        end
        simvals[i][j] = sv
        simps[i][j] = sp
    end
    vms[i] = mean(simvals[i])
    vss[i] = std(simvals[i])
    vci[i] = meanci(simvals[i])
    pms[i] = mean(simps[i])
    pss[i] = std(simps[i])
    pci[i] = meanci(simps[i])
end

ves = [policyvalue(solip, e1), policyvalue(solrip, e1),
        policyvalue(soltip, e1)]
rdata = DataFrame(ID = ["Nominal", "Robust","OffNominal"], ExpValue = ves,
            SimMean = vms, SimStd = vss, SimCI = vci,
            SimPercentMean = pms, SimPercentStd = pss, SimPercentCI = pci,
            SimProb = simprobnames)
simdata = DataFrame(NominalSim = simvals[1], RobustSim = simvals[2],
            OffNominal = simvals[3])
@show rdata

sversion = "8.1"
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnactions = string("exp_actions_", sname, "_", sversion, ".csv")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnactions), actiondata)
CSV.write(joinpath(path, fnresults), rdata)
CSV.write(joinpath(path, fnsim), simdata)

println("Data Collection Complete.")

# load packages
using RPOMDPs, RPOMDPModels, RPOMDPToolbox, RobustValueIteration
using DataFrames, Query, CSV, ProgressMeter
const rpbvi = RobustValueIteration

# setup results structures
# solutions: holds policies and full sim results
# data: holds solution values and copmutation time
factors = DataFrame(ID = Int[], Problem = String[], Short_Name = String[],
    Solution = String[], Reward = String[], Information_Function = String[],
    Uncertainty_Size = Float64[], Dynamics = String[])
probnames = ["Crying Baby", "Tiger", "SimpleBaby", "SimpleTiger"]
shortnames = ["baby", "tiger", "simplebaby", "simpletiger"]
soltypes = ["Standard", "Robust"]
rewardtypes = ["Standard", "Information"]
infofuncs = ["Simple", "Approximate Entropy"]
uncsizes = [0.001, 0.01, 0.1, 0.2]
dyntypes = ["Nominal", "Ambiguous"]
respsols = ["Policy", "Simulation Values"]
headerfacts = ["ID","Problem", "Short Name", "Solution", "Reward",
    "Information Function", "Uncertainty Size", "Dynamics"]
headersols = vcat(factors, respsols)

ind = 0
for pname in probnames, sol in soltypes, r in rewardtypes, i in infofuncs, u in uncsizes, d in dyntypes
    ind += 1
    sname = shortnames[findin(probnames,[pname])[1]]
    push!(factors, [ind, pname, sname, sol, r, i, u, d])
end

for i in 1:size(factors,1)
    if factors[:Reward][i] == "Standard"
        factors[:Information_Function][i] = "NA"
    end
    if (factors[:Solution][i] == "Standard") && (factors[:Dynamics][i] == "Nominal")
        factors[:Uncertainty_Size][i] = 0.0
    end
end
new_factors = unique(factors[:, 2:8])

# simple info reward function
const ra11 = [1.0, -1.0] # belief-reward alpha vector
const ra12 = [-1.0, 1.0] # belief-reward alpha vector
const ralphas1 = [ra11, ra12]

# complicated info reward function
const ra21 = [1.0, 0.0]
const ra22 = [0.9, 0.1]
const ra23 = [0.8, 0.2]
const ra24 = [0.7, 0.3]
const ralphas2 = [ra21, ra22, ra23, ra24, 1-ra24, 1-ra23, 1-ra22, 1-ra21]

# set up problems - WIP
function build(sname, robust, info, err, rewardfunc)
    prob = nothing
    if rewardfunc == "Simple"
        r = ralphas1
    elseif rewardfunc == "Approximate Entropy"
        r = ralphas2
    else
        r = nothing
    end
    if sname == "baby"
        if robust == "Robust"
            prob = (info == "Standard") ? BabyRPOMDP(err) : BabyInfoRPOMDP(err, r)
        else
            prob = (info == "Standard") ? BabyPOMDP() : BabyInfoPOMDP(r)
        end
    elseif sname == "maze1x3"
        if robust == "Robust"
            prob = info == "Standard" ? Maze13RPOMDP(err) : Maze13RIPOMDP(err, r)
        else
            prob = info == "Standard" ? Maze13POMDP() : Maze13IPOMDP(r)
        end
    elseif sname == "maze4x3"
        if robust == "Robust"
            prob = info == "Standard" ? Maze43RPOMDP(err) : Maze43RIPOMDP(err, r)
        else
            prob = info == "Standard" ? Maze43POMDP() : Maze43IPOMDP(r)
        end
    elseif sname == "aircraftid"
        if robust == "Robust"
            prob = info == "Standard" ? AircraftRPOMDP(err) : AircraftRIPOMDP(err, r)
        else
            prob = info == "Standard" ? AircraftPOMDP() : AircraftIPOMDP(r)
        end
    elseif sname == "paint"
        if robust == "Robust"
            prob = info == "Standard" ? PaintRPOMDP(err) :PaintRIPOMDP(err, r)
        else
            prob = info == "Standard" ? PaintPOMDP() : PaintIPOMDP(r)
        end
    elseif sname == "shuttle"
        if robust == "Robust"
            prob = info == "Standard" ? ShuttleRPOMDP(err) : ShuttleRIPOMDP(err, r)
        else
            prob = info == "Standard" ? ShuttlePOMDP() : ShuttleIPOMDP(r)
        end
    elseif sname == "tiger"
        if robust == "Robust"
            prob = info == "Standard" ? TigerRPOMDP(0.95, err) : TigerInfoRPOMDP(0.95, err, r)
        else
            prob = info == "Standard" ? TigerPOMDP(0.95) : TigerInfoPOMDP(0.95, r)
        end
    elseif sname == "simplebaby"
        if robust == "Robust"
            prob = info == "Standard" ? SimpleBaby2RPOMDP(-15.0, -10.0, 0.9, err) : BabyInfoRPOMDP(err, r)
        else
            prob = info == "Standard" ? Baby2POMDP(-15.0, -10.0, 0.9) : BabyInfoPOMDP(r)
        end
    elseif sname == "simpletiger"
        if robust == "Robust"
            prob = info == "Standard" ? SimpleTigerRPOMDP(0.95, err) : TigerInfoRPOMDP(0.95, err, r)
        else
            prob = info == "Standard" ? TigerPOMDP(0.95) : TigerInfoPOMDP(0.95, r)
        end
    end
    prob
end

##################################################

sname = "simplebaby"
sversion = "5.1"
nreps = 500

# all sname runs
# dfexp = @from run in new_factors begin
#             @where run.Short_Name == sname
#             @select run
#             @collect DataFrame
# end

# non-info sname runs
dfexp = @from run in new_factors begin
            @where run.Short_Name == sname && run.Reward == "Standard"
            @select run
            @collect DataFrame
end

nrows = size(dfexp,1)
probs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(nrows)
simprobs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(nrows)
# problems and sim problems
for i = 1:nrows
    sname = dfexp[:Short_Name][i]
    robust = dfexp[:Solution][i]
    info = dfexp[:Reward][i]
    err = dfexp[:Uncertainty_Size][i]
    rewardfunc = dfexp[:Information_Function][i]
    probs[i] = build(sname, robust, info , err, rewardfunc)
    simrobust = (dfexp[:Dynamics][i] == "Nominal") ? "Standard" : "Robust"
    simprobs[i] = build(sname, simrobust, info, err, rewardfunc)
end

bs = [[b, 1-b] for b in 0.0:0.05:1.0]
nsteps = 100
solver = RPBVISolver(beliefpoints = bs, max_iterations = nsteps)
policies = Array{AlphaVectorPolicy}(nrows)
soldynamics = Array{AlphaVectorPolicy}(nrows) # worst-case dynamics
policies[1] = RPBVI.solve(solver, probs[1])
(dfexp[:Dynamics][1] == "Ambiguous") && (soldynamics[1] = RPBVI.solve(solver, simprobs[1]))

for i = 2:nrows
    println("\rPolicy $i")
    if probs[i] == probs[i-1]
        policies[i] = policies[i-1]
    else
        policies[i] = RPBVI.solve(solver, probs[i])
    end
end

function findpolicy(simprob::Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}, probs, policies)
    ind = 0
    counter = 1
    while ind < 1
        (simprob == probs[counter]) && (ind = counter)
        counter += 1
    end
    policies[ind]
end

for i = 1:nrows
    soldynamics[i] = findpolicy(simprobs[i], probs, policies)
end

function meanci(data::Vector{Float64})
    n = length(data)
    m = mean(data)
    s = std(data)
    tstar = 1.962
    hw = tstar * s / sqrt(n)
    (m - hw, m + hw)
end

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

# rollout generates state/obs/rewards based on the sim problem, and beliefs based on the solution policy
srand(923475)
for i in 1:nrows
    println("Run $i")
    println("Solution Type: ", new_factors[:Solution][i], " Dynamics, ", new_factors[:Reward][i], " Reward")
    prob = probs[i]
    simprob = simprobs[i]
    sol = policies[i]
    ves[i] = value(sol, initial_belief(prob))
    bu = updater(sol)
    println("Simulation Type: ", new_factors[:Dynamics][i], " ", typeof(simprob))
    @showprogress 1 "Simulating value..." for j = 1:nreps
        if typeof(simprob) <: Union{RPOMDP,RIPOMDP}
            sv, sp = simulate_worst(psim, simprob, sol, bu, soldynamics[i].alphas)
        else
            sv, sp = simulate(psim, simprob, sol, bu)
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

ids = collect(1:size(dfexp,1))
dfexp[:ID] = ids
rdata = DataFrame(ID = ids, ExpectedValue = ves, SimMean = vms, SimStd = vss,
    SimCI = vci, CorrectMean = pms, CorrectStd = pss, CorrectCI = pci)
ndec = 3
rdata_round = DataFrame(ID = ids, ExpectedValue = round.(ves, ndec),
    SimMean = round.(vms, ndec),
    SimStd = round.(vss, ndec),
    SimCI = [round.(vci[i], ndec) for i in 1:length(vci)],
    CorrectMean = round.(pms, ndec),
    CorrectStd = round.(pss, ndec),
    CorrectCI = [round.(pci[i], ndec) for i in 1:length(pci)])
df = join(dfexp, rdata_round, on = :ID)
simdata = hcat(ids, hcat(simvals...)') |> DataFrame
@show rdata



path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnresults), df)
CSV.write(joinpath(path, fnsim), simdata)

using Plots; gr()
rnum = 1
valfunc1 = [maximum(dot(policies[rnum].alphas[i], [b, 1-b]) for i = 1:length(policies[rnum].alphas))
        for b = 0.0:0.01:1.0]
valfunc13 = [maximum(dot(policies[13].alphas[i], [b, 1-b]) for i = 1:length(policies[13].alphas))
        for b = 0.0:0.01:1.0]
x = collect(0:0.01:1.0)
vplot = plot(x, valfunc1, xticks = 0:0.1:1, label = "Nominal",
        # title = "Tiger POMDP Value Function",
        xlab = "Belief, P(State = Tiger Left)",
        # xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        legend = :bottomleft,
        # legend = :topright,
        line = :dash,
        linealpha = 0.9)
plot!(x, valfunc13, color = :red, linealpha = 0.8, label = "Robust: 0.2")
fn = string("value_function_", sname, "_", sversion, ".pdf")
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(vplot, path)

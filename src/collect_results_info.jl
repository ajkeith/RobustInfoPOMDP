# load packages
using RPOMDPs, RPOMDPModels, RPOMDPToolbox, RobustValueIteration
using SimpleProbabilitySets, DataFrames, Query, CSV, ProgressMeter
const rpbvi = RobustValueIteration

# setup results structures
# solutions: holds policies and full sim results
# data: holds solution values and copmutation time
factors = DataFrame(ID = Int[], Problem = String[], Short_Name = String[],
    Solution = String[], Reward = String[], Information_Function = String[],
    Uncertainty_Size = Float64[], Dynamics = String[])
probnames = ["Rock Diagnosis"]
shortnames = ["rock"]
soltypes = ["Standard", "Robust"]
rewardtypes = ["Standard", "Information"]
infofuncs = ["Simple", "Approximate Entropy"]
uncsizes = [0.001, 0.1, 0.2, 0.3, 0.4]
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
    elseif sname == "rock"
        if robust == "Robust"
            prob = info == "Standard" ? nothing : RockRIPOMDP(r, 0.95, err)
        else
            prob = info == "Standard" ? nothing : RockIPOMDP(r, 0.95)
        end
    end
    prob
end

##################################################

sname = "rock"
sversion = "8.0"
nreps = 100

# info sname runs
dfexp = @from run in new_factors begin
            @where run.Short_Name == sname && run.Reward == "Information"
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

nsteps = 150
bdelt = 0.05
nb2 = 20
bs1 = [vcat([b, 1-b],zeros(2)) for b in 0.0:bdelt:1.0]
bs2 = [vcat(zeros(2), [b, 1-b]) for b in 0.0:bdelt:1.0]
bs3 = [psample(zeros(4), ones(4)) for i = 1:nb2]
bs = vcat(bs1, bs2, bs3)
solver = RPBVISolver(beliefpoints = bs, max_iterations = nsteps)
policies = Array{AlphaVectorPolicy}(nrows)
soldynamics = Array{AlphaVectorPolicy}(nrows) # worst-case dynamics
policies[1] = rpbvi.solve(solver, probs[1], verbose = true)
(dfexp[:Dynamics][1] == "Ambiguous") && (soldynamics[1] = rpbvi.solve(solver, simprobs[1]))

# calculate policies
for i = 2:nrows
    println("\rPolicy $i")
    if dfexp[:Solution][i] == "Standard"
        if dfexp[:Information_Function][i] == dfexp[:Information_Function][i-1]
            policies[i] = policies[i-1]
        else
            policies[i] = rpbvi.solve(solver, probs[i], verbose = true)
        end
    elseif dfexp[:Solution][i] == "Robust"
        if dfexp[i, 5:6] == dfexp[i-1, 5:6]
            policies[i] = policies[i-1]
        else
            policies[i] = rpbvi.solve(solver, probs[i], verbose = true)
        end
    else
        policies[i] = rpbvi.solve(solver, probs[i], verbose = true)
    end
end

function findpolicy(dfexp, row::Int, policies)
    if dfexp[:Dynamics][row] == "Nominal"
        if dfexp[:Information_Function][row] == "Simple"
            return policies[1]
        else
            return policies[7]
        end
    else
        if dfexp[:Information_Function][row] == "Simple"
            if dfexp[:Uncertainty_Size][row] == uncsizes[1]
                return policies[13]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[2]
                return policies[15]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[3]
                return policies[17]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[4]
                return policies[19]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[5]
                return policies[21]
            else
                return nothing
            end
        else
            if dfexp[:Uncertainty_Size][row] == uncsizes[1]
                return policies[23]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[2]
                return policies[25]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[3]
                return policies[27]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[4]
                return policies[29]
            elseif dfexp[:Uncertainty_Size][row] == uncsizes[5]
                return policies[31]
            else
                return nothing
            end
        end
    end
end

for i = 1:nrows
    soldynamics[i] = findpolicy(dfexp, i, policies)
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
    binit = initial_belief_distribution(prob)
    sinit = rand(psim.rng, initial_state_distribution(simprob))
    println("Simulation Type: ", new_factors[:Dynamics][i], " ", typeof(simprob))
    @showprogress 1 "Simulating value..." for j = 1:nreps
        if typeof(simprob) <: Union{RPOMDP,RIPOMDP}
            sv, sp = simulate_worst(psim, simprob, sol, bu, binit, sinit, soldynamics[i].alphas)
        else
            sv, sp = simulate(psim, simprob, sol, bu, binit, sinit)
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
    CorrectMean = round.(pms, ndec) .* 100,
    CorrectStd = round.(pss, ndec) .* 100,
    CorrectCI = [round.(pci[i], ndec) .* 100 for i in 1:length(pci)])
dfraw = join(dfexp, rdata_round, on = :ID)
simdata = hcat(ids, hcat(simvals...)') |> DataFrame
@show rdata_round
df = sort(dfraw, [order(:Information_Function), order(:Solution),
    order(:Dynamics), order(:Uncertainty_Size)])


path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnresults), df)
CSV.write(joinpath(path, fnsim), simdata)

using Plots; gr()
rnum = 1
valfunc1 = [maximum(dot(policies[rnum].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[rnum].alphas))
        for b = 0.0:0.01:1.0]
valfunc13 = [maximum(dot(policies[13].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[24].alphas))
        for b = 0.0:0.01:1.0]
valfunc15 = [maximum(dot(policies[15].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[26].alphas))
        for b = 0.0:0.01:1.0]
valfunc17 = [maximum(dot(policies[17].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[28].alphas))
        for b = 0.0:0.01:1.0]
valfunc19 = [maximum(dot(policies[19].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[19].alphas))
        for b = 0.0:0.01:1.0]
valfunc21 = [maximum(dot(policies[21].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[21].alphas))
        for b = 0.0:0.01:1.0]
valfunc13 = [maximum(dot(policies[23].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[23].alphas))
        for b = 0.0:0.01:1.0]
valfunc15 = [maximum(dot(policies[25].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[25].alphas))
        for b = 0.0:0.01:1.0]
valfunc17 = [maximum(dot(policies[27].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[27].alphas))
        for b = 0.0:0.01:1.0]
valfunc19 = [maximum(dot(policies[29].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[29].alphas))
        for b = 0.0:0.01:1.0]
valfunc21 = [maximum(dot(policies[31].alphas[i], [b, 1-b, 0.0, 0.0]) for i = 1:length(policies[31].alphas))
        for b = 0.0:0.01:1.0]
x = collect(0:0.01:1.0)
vplot = plot(x, valfunc1, xticks = 0:0.1:1, label = "Nominal",
        # title = "Tiger POMDP Value Function",
        xlab = "Belief, P(State = Bad | Position = 1)",
        # xlab = "Belief, P(State = Hungry)",
        ylab = "Expected Total Discounted Reward",
        legend = :bottomleft,
        # legend = :topright,
        line = :dash,
        linealpha = 0.9)
plot!(x, valfunc13, color = :red, linealpha = 0.8, label = "Robust: 0.001")
plot!(x, valfunc15, color = :red, linealpha = 0.7, label = "Robust: 0.1")
plot!(x, valfunc17, color = :red, linealpha = 0.6, label = "Robust: 0.2")
plot!(x, valfunc19, color = :red, linealpha = 0.5, label = "Robust: 0.3")
plot!(x, valfunc21, color = :red, linealpha = 0.4, label = "Robust: 0.4")
fn = string("value_function_", sname, "_", sversion, ".pdf")
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(vplot, path)

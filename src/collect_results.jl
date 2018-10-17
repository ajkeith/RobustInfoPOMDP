# TO DO
# eventually move this to a jupyter lab/notebook
# write pomdp txt file for baby, babyinfo, and tigerinfo (and others?)

# load packages
using RPOMDPs, RPOMDPModels, RPOMDPToolbox, RobustValueIteration
using DataFrames, Query, CSV, ProgressMeter
const rpbvi = RobustValueIteration

# setup results structures
# solutions: holds policies and full sim results
# data: holds solution values and copmutation time
factors = DataFrame(ID = Int[], Problem = String[], Short_Name = String[], Solution = String[], Reward = String[], Information_Function = String[], Uncertainty_Size = Float64[], Dynamics = String[])
probnames = ["Crying Baby", "Maze 1x3", "Maze 4x3", "Aircraft ID", "Part Painting", "Shuttle", "Tiger"]
shortnames = ["baby", "maze1x3", "maze4x3", "aircraftid", "paint", "shuttle", "tiger"]
soltypes = ["Standard", "Robust"]
rewardtypes = ["Standard", "Information"]
infofuncs = ["Simple", "Approximate Entropy"]
uncsizes = [0.025, 0.1, 0.3]
dyntypes = ["Nominal", "Ambiguous"]
respsols = ["Policy", "Simulation Values"]
respdata = ["Solution Value", "Simulation Value (Mean)", "Simulation Value (Std Dev)", "Computation Time"]
headerfacts = ["ID","Problem", "Short Name", "Solution", "Reward", "Information Function", "Uncertainty Size", "Dynamics"]
headersols = vcat(factors, respsols)
headerdata = vcat(factors, respdata)

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
            prob = info == "Standard" ? TigerRPOMDP(err) : TigerInfoRPOMDP(err, r)
        else
            prob = info == "Standard" ? TigerPOMDP() : TigerInfoPOMDP(r)
        end
    end
    prob
end

sname = "tiger"
sversion = "3.0"
dfexp = @from run in new_factors begin
            @where run.Short_Name == sname
            @select run
            @collect DataFrame
end

nrows = size(dfexp,1)
probs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(nrows)
simprobs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(nrows)
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


# ntest = size(dfexp,1)
bs = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0]]
ntest = size(dfexp,1)
nsteps = 15
nreps = 20
solver = RPBVISolver(beliefpoints = bs, max_iterations = nsteps)
psim = RolloutSimulator(max_steps = nsteps)
soldynamics = Array{AlphaVectorPolicy}(ntest) # sim dynamics policies
policies = Array{AlphaVectorPolicy}(ntest) # solution policies
simvals = [Vector{Float64}(nreps) for _ in 1:ntest] # simulated values
simps = [Vector{Float64}(nreps) for _ in 1:ntest] # simulated percent correct
ves = Array{Float64}(ntest) # expected values
vms = Array{Float64}(ntest) # mean of sim values
vss = Array{Float64}(ntest) # std dev of sim values
pms = Array{Float64}(ntest) # mean percent correct
pss = Array{Float64}(ntest) # std percent correct

# hard coded
srand(74733)
for i in collect(14:2:30)
    soldynamics[i] = rpbvi.solve(solver, probs[i])
end
soldynamics[2:4] = soldynamics[[14,16,18]]
soldynamics[6:8] = soldynamics[[20,22,24]]
soldynamics[10:12] = soldynamics[[26,28,30]]

# rollout generates state/obs/rewards based on the sim problem, and beliefs based on the solution policy
srand(923475)
for i in 1:ntest
    println("Run $i")
    println("Solution Type: ", new_factors[:Solution][i], " Dynamics, ", new_factors[:Reward][i], " Reward")
    prob = probs[i]
    simprob = simprobs[i]
    sol = rpbvi.solve(solver, prob)
    policies[i] = sol
    ves[i] = value(sol, initial_belief(prob))
    bu = updater(sol)
    println("Simulation Type: ", new_factors[:Dynamics][i], " ", typeof(simprob))
    # simvals[i] = @showprogress 1 "Simulating value..." [simulate(psim, simprob, sol, bu) for j=1:nreps]
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
    pms[i] = mean(simps[i])
    pss[i] = std(simps[i])
end

ids = collect(1:size(dfexp,1))
dfexp[:ID] = ids
rdata = DataFrame(ID = ids, ExpectedValue = ves, SimMean = vms, SimStd = vss, CorrectMean = pms, CorrectStd = pss)
df = join(dfexp, rdata, on = :ID)
simdata = hcat(ids, hcat(simvals...)') |> DataFrame
@show rdata



path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnresults = string("exp_results_", sname, "_", sversion, ".csv")
fnsim = string("exp_sim_values_", sname, "_", sversion, ".csv")
CSV.write(joinpath(path, fnresults), df)
CSV.write(joinpath(path, fnsim), simdata)

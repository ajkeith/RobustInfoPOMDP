# TO DO
# eventually move this to a jupyter lab/notebook
# rename PBVI to RPBVI
# find out why rollout and solution value don't match
# write pomdp txt file for crying baby

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
            prob = (info == "Standard") ? BabyRPOMDP(err) : BabyRIPOMDP(err, r)
        else
            prob = (info == "Standard") ? BabyPOMDP() : BabyIPOMDP(r)
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
            prob = info == "Standard" ? TigerRPOMDP(err) : TigerRIPOMDP(err, r)
        else
            prob = info == "Standard" ? TigerPOMDP() : TigerIPOMDP(r)
        end
    end
    prob
end

dfbaby = @from run in new_factors begin
            @where run.Short_Name == "baby"
            @select run
            @collect DataFrame
end

probs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(size(dfbaby,1))
simprobs = Array{Union{POMDP,IPOMDP,RPOMDP,RIPOMDP}}(size(dfbaby,1))
for i = 1:size(dfbaby,1)
    sname = dfbaby[:Short_Name][i]
    robust = dfbaby[:Solution][i]
    info = dfbaby[:Reward][i]
    err = dfbaby[:Uncertainty_Size][i]
    rewardfunc = dfbaby[:Information_Function][i]
    probs[i] = build(sname, robust, info , err, rewardfunc)
    simrobust = (dfbaby[:Dynamics][i] == "Nominal") ? "Standard" : "Robust"
    simprobs[i] = build(sname, simrobust, info, err, rewardfunc)
end


# ntest = size(dfbaby,1)
ntest = size(dfbaby,1)
solver = PBVISolver()
nsteps = 1_000
nreps = 5
psim = RolloutSimulator(max_steps = nsteps)
policies = Array{AlphaVectorPolicy}(ntest) # solution policies
simvals = Array{Array{Float64}}(ntest) # simulated values
ves = Array{Float64}(ntest) # expected values
vms = Array{Float64}(ntest) # mean of sim values
vss = Array{Float64}(ntest) # std dev of sim values

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
    simvals[i] = @showprogress 1 "Simulating value..." [simulate(psim, simprob, sol, bu) for j=1:nreps]
    vms[i] = mean(simvals[i])
    vss[i] = std(simvals[i])
end

rstructs = (policies, simvals)
ids = collect(1:size(dfbaby,1))
dfbaby[:ID] = ids
rdata = DataFrame(ID = ids, ExpectedValue = ves, SimMean = vms, SimStd = vss)
df = join(dfbaby, rdata, on = :ID)
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
CSV.write(joinpath(path,"exp_results.csv"), df)

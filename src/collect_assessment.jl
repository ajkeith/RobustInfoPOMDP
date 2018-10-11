using RPOMDPModels, RPOMDPs, RPOMDPToolbox
using RobustValueIteration
using SimpleProbabilitySets
const RPBVI = RobustValueIteration
TOL = 1e-6

p = CyberPOMDP()
ip = CyberIPOMDP()
rp = CyberRPOMDP()
rip = CyberRIPOMDP()

nS = 27
bs = Vector{Vector{Float64}}(nS)
bs[1] = vcat(1.0, fill(0.0, nS - 1))
bs[nS] = vcat(fill(0.0, nS - 1), 1.0)
for i = 2:(nS-1)
  # bs[i] = vcat(fill(0.0, i - 1), 1.0, fill(0.0, nS - i))
  bs[i] = psample(zeros(27), ones(27))
end
push!(bs, fill(1/27, 27))
solver = RPBVISolver(beliefpoints = bs, max_iterations = 2)

solrip = RPBVI.solve(solver, rip)
solp = RPBVI.solve(solver, p)

prob = probs[5]
sol = policies[5]
e14 = zeros(27); e14[14] = 1
e15 = zeros(27); e15[15] = 1
e27 = zeros(27); e27[27] = 1
db = DiscreteBelief(prob, states(prob), e27)
action(sol, db)
policyvalue(sol, e27)


########################################
#
# Automated Data Collection
#
########################################

# load packages
using RPOMDPs, RPOMDPModels, RPOMDPToolbox, RobustValueIteration
using DataFrames, Query, CSV, ProgressMeter
const rpbvi = RobustValueIteration

# setup results structures
# solutions: holds policies and full sim results
# data: holds solution values and copmutation time
factors = DataFrame(ID = Int[], Problem = String[], Short_Name = String[], Solution = String[], Uncertainty_Size = Float64[], Dynamics = String[])
probnames = ["Assessment"]
shortnames = ["assessment"]
soltypes = ["Standard", "Robust"]
uncsizes = [0.025, 0.1, 0.3]
dyntypes = ["Nominal", "Ambiguous"]
respsols = ["Policy", "Simulation Values"]
respdata = ["Solution Value", "Simulation Value (Mean)", "Simulation Value (Std Dev)", "Computation Time"]
headerfacts = ["ID","Problem", "Short Name", "Solution", "Uncertainty Size", "Dynamics"]
headersols = vcat(factors, respsols)
headerdata = vcat(factors, respdata)

ind = 0
for pname in probnames, sol in soltypes, u in uncsizes, d in dyntypes
    ind += 1
    sname = shortnames[findin(probnames,[pname])[1]]
    push!(factors, [ind, pname, sname, sol, u, d])
end

for i in 1:size(factors,1)
    if (factors[:Solution][i] == "Standard") && (factors[:Dynamics][i] == "Nominal")
        factors[:Uncertainty_Size][i] = 0.0
    end
end
new_factors = unique(factors[:, 2:6])

# set up problems - WIP
function build(sname, robust, err)
    prob = nothing
    if sname == "assessment"
        prob = (robust == "Robust") ? CyberRIPOMDP(err) : CyberIPOMDP()
    end
    prob
end

sname = "assessment"
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
    err = dfexp[:Uncertainty_Size][i]
    probs[i] = build(sname, robust, err)
    simrobust = (dfexp[:Dynamics][i] == "Nominal") ? "Standard" : "Robust"
    simprobs[i] = build(sname, simrobust, err)
end


nS = 27
bs = Vector{Vector{Float64}}(nS)
bs[1] = vcat(1.0, fill(0.0, nS - 1))
bs[nS] = vcat(fill(0.0, nS - 1), 1.0)
for i = 2:(nS-1)
  # bs[i] = vcat(fill(0.0, i - 1), 1.0, fill(0.0, nS - i))
  bs[i] = psample(zeros(nS), ones(nS))
end
push!(bs, fill(1/nS, nS))
solver = RPBVISolver(beliefpoints = bs[19:28], max_iterations = 10)

# ntest = size(dfexp,1)
ntest = size(dfexp,1)
nreps = 3
nsteps = 10
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
    println("Solution Type: ", new_factors[:Solution][i], " Dynamics")
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

ids = collect(1:size(dfexp,1))
dfexp[:ID] = ids
rdata = DataFrame(ID = ids, ExpectedValue = ves, SimMean = vms, SimStd = vss)
df = join(dfexp, rdata, on = :ID)
simdata = hcat(ids, hcat(simvals...)') |> DataFrame
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data")
fnresults = string("exp_results_", sname, ".csv")
fnsim = string("exp_sim_values_", sname, ".csv")
CSV.write(joinpath(path, fnresults), df)
CSV.write(joinpath(path, fnsim), simdata)

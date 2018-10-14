
# compare RPOMDP to POMDP
using POMDPs, POMDPModels, POMDPToolbox
using IncrementalPruning, FIB, QMDP
using Plots; gr()

prob = TigerPOMDP()
s1 = PruneSolver(max_iterations = 20)
s2 = FIBSolver(max_iterations = 10_000)
s3 = QMDPSolver(max_iterations = 10_000)
pol1 = solve(s1, prob)
pol2 = solve(s2, prob)
pol3 = solve(s3, prob)
bu1 = updater(pol1)
bu2 = updater(pol2)
bu3 = updater(pol3)
h1 = simulate(HistoryRecorder(max_steps=100), prob, pol1, bu1)
h2 = simulate(HistoryRecorder(max_steps=100), prob, pol2, bu2)
h3 = simulate(HistoryRecorder(max_steps=100), prob, pol3, bu3)
v1 = discounted_reward(h1)
v2 = discounted_reward(h2)
v3 = discounted_reward(h3)
POMDPToolbox.test_solver(s1, prob) # 17.7
POMDPToolbox.test_solver(s2, prob) # 17.7
POMDPToolbox.test_solver(s3, prob) # 17.7
value(pol1, [0.5,0.5])
value(pol2, [0.5,0.5])
value(pol3, [0.5,0.5])
N = 1000
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1000), prob, pol1, bu1)) for i = 1:N)
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1000), prob, pol2, bu2)) for i = 1:N)
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1000), prob, pol3, bu3)) for i = 1:N)


# Incremental Pruning
# too many to plot
Plots.plot([0,1], pol1.alphas, xticks = 0:0.05:1, lab = pol1.action_map, legend = :bottomright)

# FIB
# action 1 at <0.05, action 2 at >0.95
Plots.plot([0,1], pol2.alphas, xticks = 0:0.05:1, lab = pol2.action_map, legend = :bottomright)

# AEMS
# action 1 at <0.05, action 2 at >0.95
Plots.plot([0,1], pol3.alphas, xticks = 0:0.05:1, lab = pol3.action_map, legend = :bottomright)

# different y-values for alpha-vectors
# sim value agrees for tigerpomdp discount = 0.95 is about 19.5



###############################
#
# Correctness testing
#
# TODO: Add to actual test file (i.e. take constant values from POMDP results and use that to test against in the runtests of RobustValueIteration)
################################
# compare RPOMDP to POMDP
using RPOMDPs, RPOMDPModels, RPOMDPToolbox
using RobustValueIteration
using Plots; gr()

prob = TigerPOMDP(0.95)
prob2 = TigerRPOMDP(0.95, 0.01)
sr = RPBVISolver(max_iterations = 10_000)
sr2 = RPBVISolver(max_iterations = 1000)
polr = RobustValueIteration.solve(sr, prob)
polr2 = RobustValueIteration.solve(sr2, prob2)
bur = updater(polr)
bur2 = updater(polr2)
hr = simulate(HistoryRecorder(max_steps=100), prob, polr, bur)
hr2 = simulate(HistoryRecorder(max_steps=100), prob2, polr2, bur2)
vr = discounted_reward(hr)
vr2 = discounted_reward(hr2)
value(polr, [0.5,0.5])
value(polr2, [0.5,0.5])
N = 1_000
N2 = 1
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1_000),
                                                prob, polr, bur)) for i = 1:N)
mean(discounted_reward(simulate(HistoryRecorder(max_steps=100),
                                                prob2, polr2, bur2)) for i = 1:N2)

# Robust point-based value iteration
# POMDP
Plots.plot([0,1], polr.alphas, xticks = 0:0.05:1,
            lab = polr.action_map, legend = :bottomright)

# Robust point-based value iteration
# RPOMDP
Plots.plot([0,1], polr2.alphas, xticks = 0:0.05:1,
            lab = polr2.action_map, legend = :bottomright)


prob = BabyPOMDP()
sr = RPBVISolver(max_iterations = 10_000)
polr = RobustValueIteration.solve(sr, prob)
bur = updater(polr)
hr = simulate(HistoryRecorder(max_steps=100), prob, polr, bur)
vr = discounted_reward(hr)
value(polr, [0.0, 1.0])
N = 1_000
mean(discounted_reward(simulate(HistoryRecorder(max_steps=1_000),
                                                prob, polr, bur)) for i = 1:N)

prob2 = BabyRPOMDP(0.2)
sr2 = RPBVISolver(max_iterations = 1000)
polr2 = RobustValueIteration.solve(sr2, prob2)
bur2 = updater(polr2)
hr2 = simulate(HistoryRecorder(max_steps=100), prob2, polr2, bur2)
vr2 = discounted_reward(hr2)
value(polr2, [0.0, 1.0])
N2 = 1
# mean(discounted_reward(simulate(HistoryRecorder(max_steps=100),
                                            # prob2, polr2, bur2)) for i = 1:N2)

# Robust point-based value iteration
# POMDP
Plots.plot([0,1], polr.alphas, xticks = 0:0.1:1,
        lab = polr.action_map, legend = :bottomright)

# Robust point-based value iteration
# RPOMDP
Plots.plot([0,1], polr2.alphas, xticks = 0:0.1:1,
        lab = polr2.action_map, legend = :bottomright)

using Plots; gr()

###########################################################
# Robust vs nominal performance for Rock Diagnosis
usize = [0.001, 0.1, 0.2, 0.3, 0.4]
nmean = [18.583, 18.455, 18.188, 17.996, 17.642]
nl = [18.514, 18.363, 18.061, 17.853, 17.475]
nu = [18.652, 18.547, 18.315, 18.14, 17.809]
rmean = [18.607, 18.491, 18.539, 18.463, 18.453]
rl = [18.539, 18.354, 18.426, 18.317, 18.308]
ru = [18.676, 18.628, 18.652, 18.609, 18.597]

plot(usize, nmean, label = "Nominal (Mean)",
    linecolor = :blue,
    linestyle = :dashdot,
    legend = :bottomleft,
    xlab = "Ambiguity (Half-width)",
    ylab = "Simulated Total Discounted Reward")
plot!(usize, nl, label = "Nominal (95% CI)",
    linecolor = :blue,
    linestyle = :dot)
plot!(usize, nu, label = "Nominal (95% CI)",
    linecolor = :blue,
    linestyle = :dot)
plot!(usize, rmean, label = "Robust (Mean)",
    linecolor = :red,
    linestyle = :solid)
plot!(usize, rl, label = "Robust (95% CI)",
    linecolor = :red,
    linestyle = :dash)
plot!(usize, ru, label = "Robust (95% CI)",
    linecolor = :red,
    linestyle = :dash)

fn = string("rock_robustvnominal.pdf")
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(path)
##########################################################

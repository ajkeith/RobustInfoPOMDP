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
    seriestype = :line,
    xlab = "Ambiguity (Half-width)",
    ylab = "Simulated Total Discounted Reward")
plot!(usize, nmean, seriestype = :scatter,
    marker = (5, :blue), markerstrokewidth = 0,
    label = nothing)
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

# scatter + line version
p_sl = plot(usize, [nmean, nmean, nl, nl, nu, nu, rmean, rmean, rl, rl, ru, ru],
    linecolor = [:blue :blue :blue :blue :blue :blue :red :red :red :red :red :red],
    linestyle = [:dashdot :dashdot :dot :dot :dot :dot :solid :solid :dash :dash :dash :dash],
    seriestype = [:line :scatter :line :scatter :line :scatter :line :scatter :line :scatter :line :scatter],
    linewidth = fill(2, (1, 12)),
    linealpha = fill(0.7, (1, 12)),
    markersize = hcat(fill(4, (1, 6)), fill(5, (1, 6))),
    markercolor = hcat(fill(:white, (1, 6)), fill(:red, (1, 6))),
    markeralpha = hcat(fill(1.0, (1, 6)), fill(1.0, (1, 6))),
    markerstrokewidth = hcat(fill(2, (1, 6)), fill(0, (1, 6))),
    markerstrokecolor = hcat(fill(:blue, (1, 6)), fill(:red, (1, 6))),
    markerstrokealpha = hcat(fill(0.7, (1, 6)), fill(0.0, (1, 6))),
    legend = :none,
    xlab = "Ambiguity (Half-width)",
    ylab = "Simulated Total Discounted Reward")
scatterlabels = ["Nominal (Mean)", "Nominal (95% CI)", "Robust (Mean)", "Robust (95% CI)"]
l = @layout [a b{0.25w}]
p1 = p_sl
p2 = plot(usize, [nmean, nl, rmean, rl],
        linestyle = [:dashdot :dot :solid :dash],
        linewidth = [2 2 2 2],
        linealpha = [1 1 1 1],
        linecolor = [:blue :blue :red :red],
        label=scatterlabels, grid=false, xlims=(20,3), showaxis=false)
p_sl_out = plot(p1,p2,layout=l)
fn = string("rock_robustvnominal_2.pdf")
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(p_sl_out, path)
##########################################################

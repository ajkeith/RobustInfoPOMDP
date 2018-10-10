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
solver = RPBVISolver(beliefpoints = bs)

solrip = RPBVI.solve(solver, rip)
solp = RPBVI.solve(solver, p)

function negentropy(p::AbstractArray{T}) where T<:Real
    s = zero(T)
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        if pi > z
            s += pi * log2(pi)
        end
    end
    return log2(length(p)) + s
end

using Distances, Plots; gr()
x = 0:0.01:1
y1 = [negentropy([x[i], 1 - x[i]]) for i = 1:length(x)]
y2 = [cityblock([x[i], 1 - x[i]], [0.5, 0.5]) for i = 1:length(x)]
y3 = [euclidean([x[i], 1 - x[i]], [0.5, 0.5]) for i = 1:length(x)]
plot(x, [y1, y2, y3], legend = :bottomright)

using ForwardDiff
dne(x) = ForwardDiff.gradient(negentropy, x)

ints = 0.1:0.1:0.9
n = length(ints)
alphas = Vector{Vector{Float64}}(n)
for (i, int) in enumerate(ints)
    x = [int, 1 - int]
    m = dne(x)
    y = negentropy(x)
    b = y - dot(m,x)
    a1 = dot(m, [0,1]) + b
    a2 = dot(m, [1,0]) + b
    alphas[i] = [a1, a2]
end

plot(x, y1, linecolor = :black, legend = :bottomright)
plot([0,1], alphas, linealpha = 0.2, linecolor = :black)


y

dot(m,[0,1]) + b
dot(m,[1,0]) + b

ax = [-2.32, 0.848]
plot!([0,1],ax)

using RPOMDPModels
ip = CyberIPOMDP()

rmax = -Inf
b = fill(1/27, 27)
for α in ip.inforeward
    rmax = max(rmax, dot(α, b))
end
rmax

ip.inforeward

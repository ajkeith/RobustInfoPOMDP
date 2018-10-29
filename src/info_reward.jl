################################################
# Approximate info-reward functions
################################################

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
y4 = [norm([x[i], 1 - x[i]] - [0.5, 0.5], Inf) for i = 1:length(x)]
plot(x, y1, label = "Negative Entropy", legend = :topright,
    xlab = "Belief, P(State = 1)",
    ylab = "Belief Reward",
    linestyle = :solid,
    linewidth = 1)
plot!(x, y2, label = "1-Norm", linestyle = :dash,
    linecolor = :red, linealpha = 1)
plot!(x, y3, label = "2-Norm", linestyle = :dot,
    linecolor = :red, linealpha = 0.6)
plot!(x, y4, label = "Infinity-Norm", linestyle = :dashdot,
    linecolor = :red, linealpha = 0.3)
fn = string("belief_reward_function.pdf")
path = joinpath(homedir(),".julia\\v0.6\\RobustInfoPOMDP\\data\\figures\\",fn)
savefig(path)

x = 1:0.1:1
y = 1:0.1:1
f(x,y) = negentropy([x - y])
X = repmat(x',length(y),1)
Y = repmat(y,1,length(x))
Z = map(f,X,Y)
p1 = contour(x,y,f,fill=true)
p2 = contour(x,y,Z)
plot(p1,p2)

using ForwardDiff
dne(x) = ForwardDiff.gradient(negentropy, x)

function ev(i::Int, len::Int)
    v = zeros(len)
    v[i] = 1.0
    return v
end

e1 = ev(1,4)
negentropy(e1)
negentropy(c)
negentropy([0.2, 0.25, 0.25, 0.3])
dne([0.2, 0.25, 0.25, 0.3])

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


# complicated info reward function
ra21 = [1.0, 0.0]
ra22 = [0.9, 0.1]
ra23 = [0.8, 0.2]
ra24 = [0.7, 0.3]
ralphas2 = [ra21, ra22, ra23, ra24, 1-ra24, 1-ra23, 1-ra22, 1-ra21]
plot([0,1], ralphas2)


using Plots; gr()
using Distances

function tinfo(rinfo::Vector{Vector{Float64}}, b::Vector{Float64})
    rmax = -Inf
    for α in rinfo
        rmax = max(rmax, dot(α, b))
    end
    rmax
end

# simple info reward function
irs = [[1.0, -1/3, -1/3, -1/3],
         [-1/3, 1.0, -1/3, -1/3],
         [-1/3, -1/3, 1.0, -1/3],
         [-1/3, -1/3, -1/3, 1.0]]

c = fill(1/4, 4)
e1 = ev(1,4)
tinfo(irs, c)

# complicated info reward function
vhi = -10.0
vlo = 0.1
irc = [[1.0, vhi, vhi, vhi],
        [vhi, 1.0, vhi, vhi],
        [vhi, vhi, 1.0, vhi],
        [vhi, vhi, vhi, 1.0],
        [vlo, -vlo/3, -vlo/3, -vlo/3],
        [-vlo/3, vlo, -vlo/3, -vlo/3],
        [-vlo/3, -vlo/3, vlo, -vlo/3],
        [-vlo/3, -vlo/3, -vlo/3, vlo]]

tinfo(irc, c)
tinfo(irc, e1)
tinfo(irs, [0.5, 0.5, 0.0, 0.0])
tinfo(irc, [0.5, 0.5, 0.0, 0.0])
negentropy([0.5, 0.5, 0.0, 0.0])
norm([0.5, 0.5, 0.0, 0.0] - c, 1)
x = 0:0.001:1.0
y0 = [norm([xi, 1-xi, 0.0, 0.0] - c, 1) for xi in x]
y1 = [negentropy([xi, 1-xi, 0.0, 0.0]) for xi in x]
y2 = [tinfo(irs, [xi, 1-xi, 0.0, 0.0]) for xi in x]
y3 = [tinfo(irc, [xi, 1-xi, 0.0, 0.0]) for xi in x]
plot(x, y2, label = "Simple", legend = :bottomleft,
    xlab = "Belief, P(State = Bad | Position = 1)",
    ylab = "Expected Total Discounted Reward")
plot!(x, y3, label = "Complex")
plot!(x, y2)
plot!(x, y3)

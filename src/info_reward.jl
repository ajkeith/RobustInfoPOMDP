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


# complicated info reward function
ra21 = [1.0, 0.0]
ra22 = [0.9, 0.1]
ra23 = [0.8, 0.2]
ra24 = [0.7, 0.3]
ralphas2 = [ra21, ra22, ra23, ra24, 1-ra24, 1-ra23, 1-ra22, 1-ra21]
plot([0,1], ralphas2)

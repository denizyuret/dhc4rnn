using Optim

# rosenbrock(x)=(1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
# o = optimize(rosenbrock, [0.0,0.0])
# println(o)

ndims = 10
A = randn(ndims,ndims)
H = A'*A
b = randn(ndims)
g(x)=sum(0.5*x'*H*x+x'*b)
println(optimize(g, zeros(ndims)))
# gradient: Hx+b
# hessian: H
# minimum: -H^-1 b
-H\b

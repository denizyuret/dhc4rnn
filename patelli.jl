function rosenbrock(x, g=nothing)
    n = length(x)
    f = 0.0
    for i=1:(n-1)
        f += (1-x[i])^2 + 100*(x[i+1]-x[i]^2)^2
    end
    if g != nothing
        for i=1:n
            g[i] = ((i==n ? 0 : -2*(1-x[i]) - 400*(x[i+1]-x[i]^2)*x[i]) +
                    (i==1 ? 0 : 200*(x[i]-x[i-1]^2)))
        end
    end
    return f
end

function gtest(x=randn(10);eps=1e-5)
    g = similar(x)
    @show f0 = rosenbrock(x,g)
    for i=1:length(x)
        xi = x[i]
        x[i] = xi+eps; f1 = rosenbrock(x)
        x[i] = xi-eps; f2 = rosenbrock(x)
        x[i] = xi; df = (f1-f2)/(2*eps)
        @show (i,g[i],df)
    end
end

using Base.LinAlg

function eq10(f, x0; gtol=0.95, gamma=1e-8)
    g0, x1, rb, rr, r = [ zeros(x0) for i=1:5 ]
    f0 = f(x0, g0)
    n0 = vecnorm(g0)
    nf = cosa = 0
    while cosa < gtol
        f1 = f(axpy!(gamma, randn!(r), copy!(x1,x0)))
        nf += 1
        b = f1 - f0
        axpy!(b, r, rb)
        axpy!(1, r.*r, rr)
        g1 = rb ./ (gamma * rr)
        n1 = vecnorm(g1)
        cosa = dot(g0,g1)/(n0*n1)
    end
    return nf
end

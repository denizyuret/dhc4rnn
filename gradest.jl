# Estimate gradient from a small set of points.

function gradest3(x, y)
    @assert size(x,1)==size(y,1)
    (x1,x2)=size(x)
    global dx = similar(x, (x1-1, x2))
    global dy = similar(y, (x1-1, 1))
    ix = 0
    for i=2:x1
        dx[i-1,:] = x[1,:]-x[i,:]
        dy[i-1,:] = y[1,:]-y[i,:]
    end
    dx\dy
end

function gradest2(x, y)
    @assert size(x,1)==size(y,1)
    (x1,x2)=size(x)
    x3 = div(x1*(x1-1),2)
    global dx = similar(x, (x3, x2))
    global dy = similar(y, (x3, 1))
    ix = 0
    for i=1:x1
        for j=(i+1):x1
            ix += 1
            dx[ix,:] = x[i,:]-x[j,:]
            dy[ix,:] = y[i,:]-y[j,:]
        end
    end
    dx\dy
end

function gradest0()
    for t=1:10
        x=randn(2,7)
        x[:,1]=1
        b=randn(7,1)
        y=x*b
        bb=x\y
        @show t
        display(bb')
        display(x[1,:]-x[2,:])
        display(y[1]-y[2])
    end
end

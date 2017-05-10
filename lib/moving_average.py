def avgnext(avgprev, xnext, tau):
    assert tau >= 1
    return (1-(1/tau))*avgprev + (1/tau)*xnext

def varnext(varprev, avgnext, xnext, tau):
    assert tau >= 1
    return (1-(1/tau))*varprev + (1/tau)*(xnext-avgnext)**2

def covnext(covprev, avgnext1, xnext1, avgnext2, xnext2, tau):
    assert tau >= 1
    return (1-(1/tau))*covprev + (1/tau)*(xnext1-avgnext1)*(xnext2-avgnext2)
import numpy as np

def genA(N,
    PEMPATHY = 1/3,
    PUNCOMFORTABLE = 1/3,
    PAPATHY = 1/3,

    AMEANEMPATHY = 1,
    ASTDEMPATHY = .2,
    AMEANUNCOMFORTABLE = -1,
    ASTDEVUNCOMFORTABLE = 0.2,
    AMEANAPATHETIC = 0,
    ASTDEVAPATHETIC = 0.2
    ):
    
    AMIN = AMEANUNCOMFORTABLE - 3*ASTDEVUNCOMFORTABLE
    AMAX = AMEANEMPATHY + 3*ASTDEMPATHY
    
    a = np.empty(N)
    for i in range(N):
        choice = np.random.choice(3, p = [PEMPATHY, PUNCOMFORTABLE, PAPATHY])
        if choice == 0:
            mu = AMEANEMPATHY
            sig = ASTDEMPATHY
        elif choice == 1:
            mu = AMEANUNCOMFORTABLE
            sig = ASTDEVUNCOMFORTABLE
        else:
            mu = AMEANAPATHETIC
            sig = ASTDEVAPATHETIC
        val = np.random.normal(mu, sig, 1)
        while val < AMIN or val > AMAX:
            val = np.random.normal(mu, sig, 1)
        a[i] = val
    return a

def genB(N,
    PESCALATE = 1/2,
    PDEESCALATE = 1/2,

    BMEANESCALATE = 1,
    BSTDEVESCALATE = 0.2,
    BMEANDEESCALATE = -1,
    BSTDEVDEESCALATE = 0.2,
    ):
    
    BMIN = BMEANDEESCALATE - 3*BSTDEVDEESCALATE
    BMAX = BMEANESCALATE + 3*BSTDEVESCALATE

    b = np.empty(N)
    for i in range(N):
        choice = np.random.choice(2, p = [PESCALATE, PDEESCALATE])
        if choice == 0:
            mu = BMEANESCALATE
            sig = BSTDEVESCALATE
        else:
            mu = BMEANDEESCALATE
            sig = BSTDEVDEESCALATE
        val = np.random.normal(mu, sig, 1)
        while val < BMIN or val > BMAX:
            val = np.random.normal(mu, sig, 1)
        b[i] = val
    return b

def genAlpha(N,
    ALPHAMEAN = 1,
    ALPHASTDEV = .2
    ):
    
    ALPHAMIN = max(0,ALPHAMEAN - 3*ALPHASTDEV)
    ALPHAMAX = ALPHAMEAN + 3*ALPHASTDEV
    alpha = np.empty(N)
    for i in range(N):
        mu = ALPHAMEAN
        sig = ALPHASTDEV
        val = np.random.normal(mu, sig, 1)
        while val < ALPHAMIN or val > ALPHAMAX:
            val = np.random.normal(mu, sig, 1)
        alpha[i] = val
    return alpha
    
def genBeta(N,
    BETAMEAN = 1,
    BETASTDEV = .2
    ):
    
    BETAMIN = max(0,BETAMEAN - 3*BETASTDEV)
    BETAMAX = BETAMEAN + 3*BETASTDEV
    
    beta = np.empty(N)
    for i in range(N):
        mu = BETAMEAN
        sig = BETASTDEV
        val = np.random.normal(mu, sig, 1)
        while val < BETAMIN or val > BETAMAX:
            val = np.random.normal(mu, sig, 1)
        beta[i] = val
    return beta
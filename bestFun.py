def best(pops, fits):
    bestAcc = fits[0, 0]
    best_individual = pops[0]
    for i in range(1, len(pops)):
        if fits[i, 0] > bestAcc:
            bestAcc = fits[i, 0]
            best_individual = pops[i]
        if fits[i, 0] == bestAcc:
            if len(pops[i]) < len(best_individual):
                best_individual = pops[i]

    return best_individual, bestAcc

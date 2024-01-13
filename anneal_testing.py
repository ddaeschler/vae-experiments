def kld_anneal(total_epochs, current_epoch, m, r, beta):
    tau = ((current_epoch - 1) % (total_epochs / m)) / (total_epochs / m)

    if tau > r:
        return 1.0
    else:
        return beta*tau


TOTAL_EPOCHS = 100
M = 4
R = 0.7
for epoch in range(TOTAL_EPOCHS):
    print(kld_anneal(TOTAL_EPOCHS, epoch, M, R, 0.5))

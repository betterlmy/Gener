import torch


def main():
    n_train = 50
    #
    x_train, _ = torch.sort(torch.ones(n_train) * 5)
    print(x_train.shape)
    print(x_train)
    keys = x_train.repeat((n_train, 1))
    print(keys.shape)
    print(keys)


if __name__ == "__main__":
    main()

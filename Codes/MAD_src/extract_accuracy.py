import numpy as np
import os
import argparse


def ext_acc(path1, path2, name, iteration, save_path, name_save):
    for rep in range(1, 2):
        acc_table = np.zeros(shape=(30, 1))
        acc_test = np.zeros(shape=(30, 1))
        acc_indices = np.zeros(shape=(30, 1), dtype=str)
        i = 0
        for a in [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
            for b in [0.0, 0.1, 0.01, 0.001, 0.0001]:
                acc_indices[i] = "alpha: " + str(a) + " beta: " + str(b)
                print(path1 + str(a) + "/" + str(b) + path2 + name + str(iteration) + "total_acc.npy")
                if os.path.exists(path1 + str(a) + "/" + str(b) + path2 + name + str(iteration) + "total_acc.npy"):
                    print(a, b, 0.0001)
                    accuracy = np.load(path1 + str(a) + "/" + str(b) + path2 + name + str(iteration) + "total_acc.npy")
                    accuracy_test = np.load(path1 + str(a) + "/" + str(b) + path2 + name + str(iteration) +
                                            "test_acc.npy")
                    acc_table[i] = accuracy[149]
                    acc_test[i] = accuracy_test[149]
                i += 1
        print(acc_test[:, -1])
        np.save(save_path + "/acc" + name_save + ".npy", acc_table)
        np.save(save_path + "/acc_test" + name_save + ".npy", acc_test)
        np.save(save_path + "/acc_indices" + name_save + ".npy", acc_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--path1', type=str)
    parser.add_argument('-p2', '--path2', type=str)
    parser.add_argument('-i', '--iteration', type=int)
    parser.add_argument('-n', "--name", type=str)
    parser.add_argument('-ns', "--name_saving", type=str)
    args, _ = parser.parse_known_args()

    path1 = "/share/home/fpainblanc/CNNMAD/" + args.path1 + "/Torch/"
    path2 = "/0.0001/" + args.path2
    paths = "/share/home/fpainblanc/CNNMAD/" + args.path1
    print(path1)
    print(path2)
    print(paths)

    ext_acc(path1, path2, args.name, args.iteration, paths, args.name_saving)



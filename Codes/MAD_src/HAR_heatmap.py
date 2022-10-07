import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
plt.style.use("seaborn")

"""acc_0001 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/RMT_acc/Dan2Bzh_eq256_lr0001_acc_test.npy")

end_0001 = acc_0001[:, -1].reshape(4, 3)
max_0001 = np.max(acc_0001, axis=-1).reshape(4, 3)

cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(end_0001, linewidth=1, annot=True, fmt=".4", vmin=70, vmax=100, cmap=cmap)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001])
heat_map.set_xticklabels([0.1, 0.01, 0.001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Denmark to Brittany over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()


cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(max_0001, linewidth=1, annot=True, fmt=".4", vmin=70, vmax=100, cmap=cmap)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001])
heat_map.set_xticklabels([0.1, 0.01, 0.001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Denmark to Brittany over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()


acc_0001 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/RMT_acc/Dan2Tarn_eq256_lr0001_acc_test.npy")

end_0001 = acc_0001[:, -1].reshape(4, 3)
max_0001 = np.max(acc_0001, axis=-1).reshape(4, 3)

cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(end_0001, linewidth=1, annot=True, fmt=".4", vmin=70, vmax=100, cmap=cmap)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001])
heat_map.set_xticklabels([0.1, 0.01, 0.001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Denmark to Tarn over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()


cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(max_0001, linewidth=1, annot=True, fmt=".4", vmin=70, vmax=100, cmap=cmap)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001])
heat_map.set_xticklabels([0.1, 0.01, 0.001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Denmark to Tarn over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()

acc_0001 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/RMT_acc/Dan2Aus_eq256_lr0001acc_test.npy")

end_0001 = acc_0001[:, -1].reshape(4, 3)
max_0001 = np.max(acc_0001, axis=-1).reshape(4, 3)

cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(end_0001, linewidth=1, annot=True, fmt=".4", vmin=70, vmax=100, cmap=cmap)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001])
heat_map.set_xticklabels([0.1, 0.01, 0.001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Denmark to Austria over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()


cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(max_0001, linewidth=1, annot=True, fmt=".4", vmin=70, vmax=100, cmap=cmap)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001])
heat_map.set_xticklabels([0.1, 0.01, 0.001])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Denmark to Brittany over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()"""
"""fig, axes = plt.subplots(1, 2, figsize=(10, 8))
mat = np.load("batch_eq2560target_confusion_mat.npy")
axes[0] = sns.heatmap(mat, linewidth=1, annot=True, fmt="0", vmin=1000, vmax=1000, ax=axes[0])
axes[0].title.set_text("Multiple DTW")

mat = np.load("new_one_dtw0target_confusion_mat.npy")
axes[1] = sns.heatmap(mat, linewidth=1, annot=True, fmt="0", vmin=1000, vmax=1000, ax=axes[1])
axes[1].title.set_text('Unique DTW')
plt.show()
plt.clf()
mat = np.load("batch_eq256_rep3_0target_confusion_mat.npy")
cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(5, 5))
heat_map = sns.heatmap(mat, linewidth=1, annot=True, fmt="0", vmin=1000, vmax=1000)
plt.show()"""

#  acc_hhar_5_6 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/TBZH5K_acc/acc_test_comp_0001.npy")[:, -149].reshape(6, 5)# HHAR_acc/acc_test_5_6.npy").reshape(6, 5)
#  acc_hhar_4_5 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/TBZH5K_acc/acc_test_comp_0001_rep2.npy")[:, -149].reshape(6, 5) #HHAR_acc/acc_test_4_5.npy").reshape(6, 5)
acc_hhar_5_6 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/HHAR_acc/acc_test_5_6.npy").reshape(6, 5)
acc_hhar_4_5 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/HHAR_acc/acc_test_4_5.npy").reshape(6, 5)
acc_hhar_1_3 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/HHAR_acc/acc_test_1_3.npy").reshape(6, 5)
acc_har_14_19 = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/HAR_acc/acc_test_14_19.npy").reshape(6, 5)
index = [1, 2, 3, 4, 5, 0]
index2 = [1, 2, 3, 4, 0]
acc_hhar_5_6 = acc_hhar_5_6[index]
acc_hhar_5_6 = acc_hhar_5_6[:, index2]
acc_hhar_4_5 = acc_hhar_4_5[index]
acc_hhar_4_5 = acc_hhar_4_5[:, index2]
acc_hhar_1_3 = acc_hhar_1_3[index]
acc_hhar_1_3 = acc_hhar_1_3[:, index2]
acc_har_14_19 = acc_har_14_19[index]
acc_har_14_19 = acc_har_14_19[:, index2]


# acc_proxy = np.load("/home/adr2.local/painblanc_f/PycharmProjects/MAD_CNN_test/TBZH5K_acc/perf_proxy.npy")
# print(acc_proxy.shape)
"""end_0001 = acc_0001[:, -1].reshape(6, 5)
index = [1, 2, 3, 4, 5, 0]
end_0001 = end_0001[index]
index2 = [1, 2, 3, 4, 0]
end_0001 = end_0001[:, index2]
max_0001 = np.max(acc_0001, axis=-1).reshape(6, 5)

end_001 = acc_001[:, -1].reshape(6, 5)
end_001 = end_001[index]
#  end_001 = np.concatenate((end_001[1:], end_001[0]))
max_001 = np.max(acc_001, axis=-1).reshape(6, 5)"""

"""cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(6, 5))
heat_map = sns.heatmap(acc_proxy[:, :, 0, 0], linewidth=1, annot=True, fmt=".4", vmin=80, vmax=100, cmap=cmap)
# , annot=True, yticklabels=ylabels, xticklabels=False)
heat_map.set_yticklabels([0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001])
heat_map.set_xticklabels([0.0, 0.1, 0.01, 0.001, 0.0001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Britains proxys labels over 10000 iterations, learning rate 0.0001, batchsize 256')
plt.show()
plt.clf()"""

"""cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(6, 5))
heat_map = sns.heatmap(acc_proxy[:, :, 1, 0], linewidth=1, annot=True, fmt=".4", vmin=80, vmax=100, cmap=cmap)
# , annot=True, yticklabels=ylabels, xticklabels=False)
heat_map.set_yticklabels([0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001])
heat_map.set_xticklabels([0.0, 0.1, 0.01, 0.001, 0.0001])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('Hyperparameters on Britains proxys labels over 10000 iterations, learning rate 0.001, batchsize 256')
#  plt.show()
plt.clf()"""

cmap = sns.cm.rocket_r
fig = plt.figure(figsize=(6, 5))
heat_map = sns.heatmap(acc_hhar_5_6, linewidth=1, annot=True, fmt=".4", vmin=50, vmax=100, cmap=cmap)
# , annot=True, yticklabels=ylabels, xticklabels=False)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0])
heat_map.set_xticklabels([0.1, 0.01, 0.001, 0.0001, 0.0])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('HHAR 5 to 6. Codats : 90.7, Codats-WS 91.7')
plt.show()
plt.clf()


fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(acc_hhar_4_5, linewidth=1, annot=True, fmt=".4", vmin=50, vmax=100, cmap=cmap)
# , annot=True, yticklabels=ylabels, xticklabels=False)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0])
heat_map.set_xticklabels([0.1, 0.01, 0.001, 0.0001, 0.0])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('HHAR 4 to 5. Codats : 94.2 Codats-WS : 94.7')
plt.show()
plt.clf()


fig = plt.figure(figsize=(4, 3))
heat_map = sns.heatmap(acc_hhar_1_3, linewidth=1, annot=True, fmt=".4", vmin=50, vmax=100, cmap=cmap)
# , annot=True, yticklabels=ylabels, xticklabels=False)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0])
heat_map.set_xticklabels([0.1, 0.01, 0.001, 0.0001, 0.0])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('HHAR 1 to 3. Codats : 93.2 Codats-WS : 90.8')
plt.show()
plt.clf()


fig = plt.figure(figsize=(6, 5))
heat_map = sns.heatmap(acc_har_14_19, linewidth=1, annot=True, fmt=".4", vmin=50, vmax=100, cmap=cmap)
# , annot=True, yticklabels=ylabels, xticklabels=False)
heat_map.set_yticklabels([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0])
heat_map.set_xticklabels([0.1, 0.01, 0.001, 0.0001, 0.0])  # , 0.001, 0.0001, 0.00001, 0.0])
plt.xlabel("Beta")
plt.ylabel('Alpha')
plt.title('HAR 14 to 19, CODATS : 72.2, CoDTAS-WS : 98.6')
# plt.show()
plt.clf()



import pickle
import matplotlib.pyplot as plt
import numpy as np

names_of_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_file(path):
    with open(path, 'rb') as fo:
        temp = pickle.load(fo, encoding='latin1')
    return temp

def visualize_samples(dataset):
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.6)

    for i in range(10):
        count = 0
        for j in range(len(dataset[0])):
            if dataset[0][j] == i + 1:
                fig.add_subplot(10, 5, 5 * i + count + 1)
                plt.imshow(dataset[1][j].reshape(3, 32, 32).transpose(1, 2, 0))
                plt.title(names_of_labels[dataset[0][j]])

                count += 1
                if count == 5: break
    # fig.tight_layout()
    plt.show()
    


dataset = [] # [label][data]

for i in range(1):
    dataset.append(load_file('./2020094_OjusSinghal_Assignment3/cifar-10-batches-py/data_batch_' + str(i + 1))['labels'])
    dataset.append(load_file('./2020094_OjusSinghal_Assignment3/cifar-10-batches-py/data_batch_' + str(i + 1))['data'])

visualize_samples(dataset)
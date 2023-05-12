import matplotlib.pyplot as plt

results_path = "/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/Results"

def plot_loss(history):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+'/augtrain_van_model_loss.png'
    plt.savefig(plot_path)
    
def plot_accuracy(history):
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plot_path = results_path+'/augtrain_van_model_accuracy.png'
    plt.savefig(plot_path)
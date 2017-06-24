import os
import shutil

current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')

data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")

# Get the dataset if it is missing
def DownloadDataset(url, path):
    import requests, zipfile, StringIO
    print "Downloading... ", url, " to ", path
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)

def GenerateDB(image, label, name):
    name = os.path.join(data_folder, name)
    print 'DB: ', name
    if not os.path.exists(name):
        syscall = "/usr/local/bin/make_mnist_db --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
        # print "Creating database with: ", syscall
        os.system(syscall)
    else:
        print "Database exists already. Delete the folder if you have issues/corrupted DB, then rerun this."
        if os.path.exists(os.path.join(name, "LOCK")):
            # print "Deleting the pre-existing lock file"
            os.remove(os.path.join(name, "LOCK"))

def PrepareDataset():
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(label_file_train):
        DownloadDataset("https://download.caffe2.ai/datasets/mnist/mnist.zip", data_folder)
    
    if os.path.exists(root_folder):
        print("Looks like you ran this before, so we need to cleanup those old files...")
        shutil.rmtree(root_folder)
    
    os.makedirs(root_folder)
    #workspace.ResetWorkspace(root_folder)

    # (Re)generate the leveldb database (known to get corrupted...) 
    GenerateDB(image_file_train, label_file_train, "mnist-train-nchw-leveldb")
    GenerateDB(image_file_test, label_file_test, "mnist-test-nchw-leveldb")

    print("training data folder:" + data_folder)
    #print("workspace root folder:" + root_folder)


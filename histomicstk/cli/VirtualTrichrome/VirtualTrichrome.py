import os, zipfile, json
from histomicstk.cli.utils import CLIArgumentParser
from glob import glob

def main(args):

    def get_base_model_name(model_file):
        try:
            base_model = model_file.split('.meta')
            assert len(base_model) == 2
            base_model = base_model[0]
        except:
            try:
                base_model = model_file.split('.index')
                assert len(base_model) == 2
                base_model = base_model[0]
            except:
                try:
                    base_model = model_file.split('.data')
                    assert len(base_model) == 2
                    base_model = base_model[0]
                except:
                    base_model = model_file
        return base_model

    cwd = os.getcwd()
    print(cwd)

    print('\n>> CLI Parameters ...\n')

    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')
    
    print("lr: " + str(args.lr))
    print("epochs: " + str(args.epochs))
    print("checkpoint_freq: " + str(args.checkpoint_freq))
    print("modelsave_freq: " + str(args.modelsave_freq))
    print("batchsize: " + str(args.batchsize))
    print("lamda: " + str(args.lamda))
    print("model: " + str(args.model))
    print("dataroot: " + str(args.dataroot))
    print("experiment_id: " + str(args.experiment_id))
    print("optimizer: " + str(args.optimizer))
    print("monitor_freq: " + str(args.monitor_freq))

    print('\n>> Output Directory Prints ...\n')

    tmp = args.outputAnnotationFile
    tmp = os.path.dirname(tmp)
    print(tmp)

    # move to data folder and extract models
    os.chdir(tmp)
    # unpck model files from zipped folder
    with open(args.inputModelFile, 'rb') as fh:
        z = zipfile.ZipFile(fh)
        for name in z.namelist():
            z.extract(name, tmp)
    
    print("\n>>List TMP directory with all files in\n")
    print(os.listdir(tmp))
    # get num_classes from json file
    with open('args.txt', 'rb') as file:
        trainingDict = json.load(file)

    print("\n>>Training Dictionary\n")
    print(trainingDict)

    epochs = trainingDict['epochs']
    lr = trainingDict['lr']

    # move back to cli folder
    os.chdir(cwd)

    model_files = glob('{}/*.h5*'.format(tmp))
    print(model_files)
    model_file = model_files[0]
    model = get_base_model_name(model_file)

    # list files code can see
    # os.system('ls -l {}'.format(model.split('model.ckpt')[0]))

    # run vis.py with flags
    cmd = "python3 ../virtualtrichrome/runWsiTest.py --model {} --input {} --output {}".format(model, args.inputImageFile, args.outputVirtualSlideImage)
    os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())

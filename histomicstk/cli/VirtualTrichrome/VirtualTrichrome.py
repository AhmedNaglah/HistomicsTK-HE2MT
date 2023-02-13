import json
import logging
import os
import time

import large_image
import numpy as np

import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.utils as htk_utils
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.CRITICAL)

def main(args):
    import dask

    total_start_time = time.time()

    print('\n>> CLI Parameters ...\n')

    print(args)

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')
    
    print("lr" + str(args.lr))
    print("epochs" + str(args.epochs))
    print("checkpoint_freq" + str(args.checkpoint_freq))
    print("modelsave_freq" + str(args.modelsave_freq))
    print("batchsize" + str(args.batchsize))
    print("lamda" + str(args.lamda))
    print("model" + str(args.model))
    print("dataroot" + str(args.dataroot))
    print("experiment_id" + str(args.experiment_id))
    print("optimizer" + str(args.optimizer))
    print("monitor_freq" + str(args.monitor_freq))

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    start_time = time.time()

    c = cli_utils.create_dask_client(args)

    print(c)

    dask_setup_time = time.time() - start_time
    print('Dask setup time = {}'.format(
        cli_utils.disp_time_hms(dask_setup_time)))

    #
    # Read Input Image
    #
    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile)

    ts_metadata = ts.getMetadata()

    print(json.dumps(ts_metadata, indent=2))


if __name__ == "__main__":

    main(CLIArgumentParser().parse_args())

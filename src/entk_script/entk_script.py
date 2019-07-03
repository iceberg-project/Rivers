"""
Seals Use Case EnTK Analysis Script
==========================================================

This script contains the EnTK Pipeline script for the Rivers   Use Case

Author: Ioannis Paraskevakos
License: MIT
Copyright: 2018-2019
"""

from __future__ import print_function
import argparse
import os
import pandas as pd

from radical.entk import Pipeline, Stage, Task, AppManager


def generate_discover_pipeline(path):
    '''
    This function takes as an input a path on Bridges and returns a pipeline
    that will provide a file for all the images that exist in that path.
    '''
    pipeline = Pipeline()
    pipeline.name = 'Disc'
    stage = Stage()
    stage.name = 'Disc-S0'
    # Create Task 1, training
    task = Task()
    task.name = 'Disc-T0'
    task.pre_exec = ['module load anaconda3/2019.03',
                     'source activate keras-gpu']
    task.executable = 'python3'   # Assign executable to the task
    task.arguments = ['image_disc.py', '%s' % path, '--filename=images.csv',
                      '--filesize']
    task.download_output_data = ['images.csv']
    task.upload_input_data = ['image_disc.py']
    task.cpu_reqs = {'processes': 1, 'threads_per_process': 1,
                     'thread_type': 'OpenMP'}
    stage.add_tasks(task)
    # Add Stage to the Pipeline
    pipeline.add_stages(stage)

    return pipeline


def generate_pipeline(name, image, image_size, model_name, device):

    '''
    This function creates a pipeline for an image that will be analyzed.

    :Arguments:
        :name: Pipeline name, str
        :image: image path, str
        :image_size: image size in MBs, int
        :model_name: Prediction Model Name, str
        :device: Which GPU device will be used by this pipeline, int
    '''
    # Create a Pipeline object
    entk_pipeline = Pipeline()
    entk_pipeline.name = name
    # Create a Stage object
    stage0 = Stage()
    stage0.name = '%s-S0' % (name)
    # Create Task 1, training
    task0 = Task()
    task0.name = '%s-T0' % stage0.name
    task0.pre_exec = ['module load matlab']
    task0.executable = 'matlab'   # Assign executable to the task
    # Assign arguments for the task executable
    task0.arguments = ["-nodisplay", "-nosplash", "-r",
                       "multipagetiff('%s','$NODE_LFS_PATH/%s');exit"
                       % (image.split('/')[-1], task0.name)] 
    task0.upload_input_data = [os.path.abspath('../utils/multipagetiff.m'),
                               os.path.abspath('../utils/saveastiff.m')]
    task0.link_input_data = [image]
    task0.cpu_reqs = {'processes': 1, 'threads_per_process': 1,
                      'thread_type': 'OpenMP'}
    task0.lfs_per_process = image_size

    stage0.add_tasks(task0)
    # Add Stage to the Pipeline
    entk_pipeline.add_stages(stage0)

    # Create a Stage object
    stage1 = Stage()
    stage1.name = '%s-S1' % (name)
    # Create Task 1, training
    task1 = Task()
    task1.name = '%s-T1' % stage1.name
    task1.pre_exec = ['module load anaconda3/2019.03',
                      'source activate keras-gpu'
                      'export CUDA_VISIBLE_DEVICES=%d' % device]
    task1.executable = 'python3'   # Assign executable to the task

    # Assign arguments for the task executable
    task1.arguments = ['predict.py',
                       '--input', '$NODE_LFS_PATH/%s/multi-' % (task0.name, image.split('/')[-1]),
                       '--output_folder', '$NODE_LFS_PATH/%s' % task1.name]
    task1.link_input_data = ['$SHARED/unet_weights.hdf5']
    task1.upload_input_data = [os.path.abspath('../classification/predict.py'),
                               os.path.abspath('../classification/' +
                                               'gen_patches.py'),
                               os.path.abspath('../classification/' +
                                               'train_unet.py'),
                               os.path.abspath('../classification/' +
                                               'unet_model.py')]
    task1.cpu_reqs = {'processes': 1, 'threads_per_process': 1,
                      'thread_type': 'OpenMP'}
    task1.gpu_reqs = {'processes': 1, 'threads_per_process': 1,
                      'thread_type': 'OpenMP'}
    task1.tag = task0.name
#
    #stage1.add_tasks(task1)
    ## Add Stage to the Pipeline
    #entk_pipeline.add_stages(stage1)

    return entk_pipeline


def args_parser():

    '''
    Argument Parsing Function for the script.
    '''
    parser = argparse.ArgumentParser(description='Executes the Seals ' +
                                     'pipeline for a set of images')

    parser.add_argument('-c', '--cpus', type=int, default=1,
                        help='The number of CPUs required for execution')
    parser.add_argument('-g', '--gpus', type=int, default=1,
                        help='The number of GPUs required for execution')
    parser.add_argument('-ip', '--input_dir', type=str,
                        help='Images input directory on the selected resource')
    parser.add_argument('-m', '--model', type=str,
                        help='Which model will be used')
    parser.add_argument('-p', '--project', type=str,
                        help='The project that will be charged')
    parser.add_argument('-q', '--queue', type=str,
                        help='The queue from which resources are requested.')
    parser.add_argument('-r', '--resource', type=str,
                        help='HPC resource on which the script will run.')
    parser.add_argument('-w', '--walltime', type=int,
                        help='The amount of time resources are requested in' +
                        ' minutes')
    parser.add_argument('--name', type=str,
                        help='name of the execution. It has to be a unique' +
                        ' value')

    return parser.parse_args()


if __name__ == '__main__':

    args = args_parser()

    res_dict = {'resource': args.resource,
                'walltime': args.walltime,
                'cpus': args.cpus,
                'gpus': args.gpus,
                'schema': 'gsissh',
                'project': args.project,
                'queue': args.queue}

    try:

        # Create Application Manager
        appman = AppManager(port=32773, hostname='localhost', name=args.name,
                            autoterminate=False, write_workflow=True)

        # Assign resource manager to the Application Manager
        appman.resource_desc = res_dict
        appman.shared_data = [os.path.abspath('../../models/unet_weights.hdf5')]
        # Create a task that discovers the dataset
        disc_pipeline = generate_discover_pipeline(args.input_dir)
        appman.workflow = set([disc_pipeline])

        # Run
        appman.run()
        print('Run Discovery')
        images = pd.read_csv('images.csv')
        
        print('Images Found:', len(images))
        # Create a single pipeline per image
        pipelines = list()
        dev = 0
        for idx in range(0,len(images)):
            p1 = generate_pipeline(name='P%s' % idx,
                                   image=images['Filename'][idx],
                                   image_size=images['Size'][idx],
                                   model_name='test',
                                   device=dev)
            dev = dev ^ 1
            pipelines.append(p1)
        # Assign the workflow as a set of Pipelines to the Application Manager
        appman.workflow = set(pipelines)

        # Run the Application Manager
        appman.run()

        print('Done')

    finally:
        # Now that all images have been analyzed, release the resources.
        print('Closing resources')
        appman.resource_terminate()

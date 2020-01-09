# modified from https://github.com/anderskm/exportTensorFlowLog/blob/master/exportTensorFlowLog.py

# import tensorflow as tf
import time
import csv
import sys
import os
import collections

import glob
import numpy as np

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False;
# TF version < 1.1.0
if (not eventAccumulatorImported):
    try:
        from tensorflow.python.summary import event_accumulator
        eventAccumulatorImported = True;
    except ImportError:
        eventAccumulatorImported = False;
# TF version = 1.1.0
if (not eventAccumulatorImported):
    try:
        from tensorflow.tensorboard.backend.event_processing import event_accumulator
        eventAccumulatorImported = True;
    except ImportError:
        eventAccumulatorImported = False;
# TF version >= 1.3.0
if (not eventAccumulatorImported):
    try:
        from tensorboard.backend.event_processing import event_accumulator
        eventAccumulatorImported = True;
    except ImportError:
        eventAccumulatorImported = False;
# TF version = Unknown
if (not eventAccumulatorImported):
    raise ImportError('Could not locate and import Tensorflow event accumulator.')

summariesDefault = ['scalars'] # ,'histograms','images','audio','compressedHistograms'];

class Timer(object):
    # Source: https://stackoverflow.com/a/5849861
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
            print('Elapsed: %s' % (time.time() - self.tstart))

def exitWithUsage():
    print(' ');
    print('Usage:');
    print('   python readLogs.py <output-folder> <output-path-to-csv> <summaries>');
    print('Inputs:');
    print('   <input-path-to-logfile>  - Path to TensorFlow logfile.');
    print('   <output-folder>          - Path to output folder.');
    print('   <summaries>              - (Optional) Comma separated list of summaries to save in output-folder. Default: ' + ', '.join(summariesDefault));
    print(' ');
    sys.exit();

def convert(inputLogFile, outputFolder, summaries):
    print(' ');
    print('> Log file: ' + inputLogFile);
    print('> Output folder: ' + outputFolder);
    print('> Summaries: ' + ', '.join(summaries));

    if any(x not in summariesDefault for x in summaries):
        print('Unknown summary! See usage for acceptable summaries.');
        exitWithUsage();


    print(' ');
    print('Setting up event accumulator...');
    with Timer():
        ea = event_accumulator.EventAccumulator(inputLogFile,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
        })

    print(' ');
    print('Loading events from file*...');
    print('* This might take a while. Sit back, relax and enjoy a cup of coffee :-)');
    with Timer():
        ea.Reload() # loads events from file

    print(' ');
    print('Log summary:');
    tags = ea.Tags();
    for t in tags:
        tagSum = []
        if (isinstance(tags[t],collections.Sequence)):
            tagSum = str(len(tags[t])) + ' summaries';
        else:
            tagSum = str(tags[t]);
        print('   ' + t + ': ' + tagSum);

    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder);

    if ('audio' in summaries):
        print(' ');
        print('Exporting audio...');
        with Timer():
            print('   Audio is not yet supported!');

    if ('compressedHistograms' in summaries):
        print(' ');
        print('Exporting compressedHistograms...');
        with Timer():
            print('   Compressed histograms are not yet supported!');


    if ('histograms' in summaries):
        print(' ');
        print('Exporting histograms...');
        with Timer():
            print('   Histograms are not yet supported!');

    if ('images' in summaries):
        print(' ');
        print('Exporting images...');
        imageDir = outputFolder + 'images'
        print('Image dir: ' + imageDir);
        with Timer():
            imageTags = tags['images'];
            for imageTag in imageTags:
                images = ea.Images(imageTag);
                imageTagDir = imageDir + '/' + imageTag;
                if not os.path.isdir(imageTagDir):
                    os.makedirs(imageTagDir);
                for image in images:
                    imageFilename = imageTagDir + '/' + str(image.step) + '.png';
                    with open(imageFilename,'wb') as f:
                        f.write(image.encoded_image_string);

    if ('scalars' in summaries):
        print(' ');
        csvFileName =  os.path.join(outputFolder,'tensorboard_log.csv');
        print('Exporting scalars to csv-file...');
        print('   CSV-path: ' + csvFileName);
        scalarTags = tags['scalars'];
        with Timer():
            with open(csvFileName,'w') as csvfile:
                logWriter = csv.writer(csvfile, delimiter=',');

                # Write headers to columns
                # headers = ['wall_time','step'];
                # for s in scalarTags:
                headers = scalarTags[:]
                # headers.append(s);
                logWriter.writerow(headers)

                max_step = 0
                for s in scalarTags:
                    vals = ea.Scalars(s)
                    for i in range(len(vals)):
                        S = vals[i].step
                        max_step = max(S, max_step)

                num_tags = len(scalarTags)
                num_steps = max_step + 1 # assume 0 is a step
                D = np.empty((num_steps, num_tags))
                D[:] = np.nan

                for j, s in enumerate(scalarTags):
                    vals = ea.Scalars(s)
                    for i in range(len(vals)):
                        S = vals[i].step
                        V = vals[i].value
                        W = vals[i].wall_time
                        D[S, j] = V

                for i in range(len(D)):
                    data = D[i, :]
                    row = [None if np.isnan(v) else v for v in data]
                    logWriter.writerow(row)

                # for i in range(len(vals)):
                #     v = vals[i];
                #     data = [v.wall_time, v.step];
                #     for s in scalarTags:
                #         scalarTag = ea.Scalars(s);
                #         S = scalarTag[i];
                #         data.append(S.value);
                #     logWriter.writerow(data);

    print(' ');
    print('Bye bye...');

if __name__ == "__main__":

    # if (len(sys.argv) < 3):
    #     exitWithUsage();

    # inputLogFile = sys.argv[1];
    # outputFolder = sys.argv[2];

    # if (len(sys.argv) < 4):
    #     summaries = summariesDefault;
    # else:
    #     if (sys.argv[3] == 'all'):
    #         summaries = summariesDefault;
    #     else:
    #         summaries = sys.argv[3].split(',');

    # convert(inputLogFile, outputFolder, summaries)

    inputFolder = sys.argv[1];

    folders = glob.glob(inputFolder + "/**/tensorboard/", recursive=True)

    for f in folders:
        inputLogFile = f
        outputFolder = f + "../"
        summaries = summariesDefault
        # try:
        convert(inputLogFile, outputFolder, summaries)
        # except:
            # print("failed at", inputLogFile)

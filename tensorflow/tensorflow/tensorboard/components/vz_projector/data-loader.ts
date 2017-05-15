/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {runAsyncTask} from './util';
import * as logging from './logging';
import {ColumnStats, DataPoint, DataSet, DatasetMetadata, MetadataInfo, PointMetadata, State, DataProto} from './data';


/** Maximum number of colors supported in the color map. */
const NUM_COLORS_COLOR_MAP = 20;

const METADATA_MSG_ID = 'metadata';
const TENSORS_MSG_ID = 'tensors';

/** Information associated with a tensor. */
export interface TensorInfo {
  /** Name of the tensor. */
  name: string;
  /** The shape of the tensor. */
  shape: [number, number];
  /** The path to the metadata file associated with the tensor. */
  metadataFile: string;
  /** The path to the bookmarks file associated with the tensor. */
  bookmarksFile: string;
}

/** Information for the model checkpoint. */
export interface CheckpointInfo {
  tensors: {[name: string]: TensorInfo};
  checkpointFile: string;
}

export type ServingMode = 'demo' | 'server' | 'proto';

/** Interface between the data storage and the UI. */
export interface DataProvider {
  /** Returns a list of run names that have embedding config files. */
  retrieveRuns(callback: (runs: string[]) => void): void;

  /**
   * Returns info about the checkpoint: number of tensors, their shapes,
   * and their associated metadata files.
   */
  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void): void;

  /** Fetches and returns the tensor with the specified name. */
  retrieveTensor(run: string, tensorName: string, callback: (ds: DataSet) => void);

  /**
   * Fetches the metadata for the specified tensor.
   */
  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void): void;

  /**
   * Returns the name of the tensor that should be fetched by default.
   * Used in demo mode to load a tensor when the app starts. Returns null if no
   * default tensor exists.
   */
  getDefaultTensor(run: string, callback: (tensorName: string) => void): void;

  getBookmarks(run: string, tensorName: string, callback: (r: State[]) => void):
      void;
}

/**
 * Data provider that loads data provided by a python server (usually backed
 * by a checkpoint file).
 */
class ServerDataProvider implements DataProvider {
  private routePrefix: string;
  private runCheckpointInfoCache: {[run: string]: CheckpointInfo} = {};

  constructor(routePrefix: string) {
    this.routePrefix = routePrefix;
  }

  retrieveRuns(callback: (runs: string[]) => void): void {
    let msgId = logging.setModalMessage('Fetching runs...');
    d3.json(`${this.routePrefix}/runs`, (err, runs) => {
      logging.setModalMessage(null, msgId);
      callback(runs);
    });
  }

  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void)
      : void {
    if (run in this.runCheckpointInfoCache) {
      callback(this.runCheckpointInfoCache[run]);
      return;
    }

    let msgId = logging.setModalMessage('Fetching checkpoint info...');
    d3.json(`${this.routePrefix}/info?run=${run}`, (err, checkpointInfo) => {
      logging.setModalMessage(null, msgId);
      this.runCheckpointInfoCache[run] = checkpointInfo;
      callback(checkpointInfo);
    });
  }

  retrieveTensor(run: string, tensorName: string, callback: (ds: DataSet) => void) {
    // Get the tensor.
    logging.setModalMessage('Fetching tensor values...', TENSORS_MSG_ID);
    d3.text(
        `${this.routePrefix}/tensor?run=${run}&name=${tensorName}`,
        (err: Error, tsv: string) => {
          if (err) {
            console.error(err);
            return;
          }
          parseTensors(tsv).then(dataPoints => {
            callback(new DataSet(dataPoints));
          });
        });
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void) {
    logging.setModalMessage('Fetching metadata...', METADATA_MSG_ID);
    d3.text(
        `${this.routePrefix}/metadata?run=${run}&name=${tensorName}`,
        (err: Error, rawMetadata: string) => {
          if (err) {
            console.error(err);
            return;
          }
          parseMetadata(rawMetadata).then(result => callback(result));
        });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    this.retrieveCheckpointInfo(run, checkpointInfo => {
      let tensorNames = Object.keys(checkpointInfo.tensors);
      // Return the first tensor that has metadata.
      for (let i = 0; i < tensorNames.length; i++) {
        let tensorName = tensorNames[i];
        if (checkpointInfo.tensors[tensorName].metadataFile) {
          callback(tensorName);
          return;
        }
      }
      callback(tensorNames.length >= 1 ? tensorNames[0] : null);
    });
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    let msgId = logging.setModalMessage('Fetching bookmarks...');
    d3.json(
        `${this.routePrefix}/bookmarks?run=${run}&name=${tensorName}`,
        (err, bookmarks) => {
          logging.setModalMessage(null, msgId);
          if (!err) {
            callback(bookmarks as State[]);
          }
        });
  }
}

class ProtoDataProvider implements DataProvider {
  private dataProto: DataProto;

  constructor(dataProto: DataProto) {
    this.dataProto = dataProto;
  }

  retrieveRuns(callback: (runs: string[]) => void): void {
    callback(['proto']);
  }

  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void) {
    callback({
      tensors: {
        'proto': {
          name: 'proto',
          shape: this.dataProto.shape,
          metadataFile: 'proto',
          bookmarksFile: null
        }
      },
      checkpointFile: 'proto'
    });
  }

  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void) {
    callback(this.flatArrayToDataset(this.dataProto.tensor));
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void): void {
    let columnNames = this.dataProto.metadata.columns.map(c => c.name);
    let n = this.dataProto.shape[0];
    let pointsMetadata: PointMetadata[] = new Array(n);
    this.dataProto.metadata.columns.forEach(c => {
      let values = c.numericValues || c.stringValues;
      for (let i = 0; i < n; i++) {
        pointsMetadata[i] = pointsMetadata[i] || {};
        pointsMetadata[i][c.name] = values[i];
      }
    });
    callback({
      stats: analyzeMetadata(columnNames, pointsMetadata),
      pointsInfo: pointsMetadata
    });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void): void {
    callback('proto');
  }

  getBookmarks(run: string, tensorName: string,
      callback: (r: State[]) => void): void {
    return callback([]);
  }

  private flatArrayToDataset(tensor: number[]): DataSet {
    let points: DataPoint[] = [];
    let n = this.dataProto.shape[0];
    let d = this.dataProto.shape[1];
    if (n * d !== tensor.length) {
      throw 'The shape doesn\'t match the length of the flattened array';
    }
    for (let i = 0; i < n; i++) {
      let vector: number[] = [];
      let offset = i * d;
      for (let j = 0; j < d; j++) {
        vector.push(tensor[offset++]);
      }
      points.push({
        vector: vector,
        metadata: {},
        projections: null,
        projectedPoint: null,
        index: i
      });
    }
    return new DataSet(points);
  }
}

/**
 * Returns a data provider, depending on what is available. The detection of
 * a server backend is done by issuing an HTTP request at /data/info and seeing
 * if it returns 200 or 404.
 *
 * @param servingMode Information how the data served (server, proto, etc.).
 * @param dataProto The projector data, in a proto format. Available if
 *     serving mode is 'proto'.
 * @param routePrefix The prefix to add to the url routes when asking for data
 *     from the backend. For example, when hosted inside tensorboard, the route
 *     is prefixed by the plugin name.
 * @param callback Called with the data provider.
 */
export function getDataProvider(servingMode: ServingMode, dataProto: DataProto,
    routePrefix: string, callback: (dp: DataProvider) => void) {
  if (servingMode === 'demo') {
    callback(new DemoDataProvider());
  } else if (servingMode === 'server') {
    if (!routePrefix) {
      throw 'route-prefix is a required parameter';
    }
    callback(new ServerDataProvider(routePrefix));
  } else if (servingMode === 'proto' && dataProto != null) {
    callback(new ProtoDataProvider(dataProto));
  }
}

export function parseRawTensors(
    content: string, callback: (ds: DataSet) => void) {
  parseTensors(content).then(data => {
    callback(new DataSet(data));
  });
}

export function parseRawMetadata(
    contents: string, callback: (r: MetadataInfo) => void) {
  parseMetadata(contents).then(result => callback(result));
}

/** Parses a tsv text file. */
function parseTensors(content: string, delim = '\t'): Promise<DataPoint[]> {
  let data: DataPoint[] = [];
  let numDim: number;
  return runAsyncTask('Parsing tensors...', () => {
    let lines = content.split('\n');
    lines.forEach(line => {
      line = line.trim();
      if (line === '') {
        return;
      }
      let row = line.split(delim);
      let dataPoint: DataPoint = {
        metadata: {},
        vector: null,
        index: data.length,
        projections: null,
        projectedPoint: null
      };
      // If the first label is not a number, take it as the label.
      if (isNaN(row[0] as any) || numDim === row.length - 1) {
        dataPoint.metadata['label'] = row[0];
        dataPoint.vector = row.slice(1).map(Number);
      } else {
        dataPoint.vector = row.map(Number);
      }
      data.push(dataPoint);
      if (numDim == null) {
        numDim = dataPoint.vector.length;
      }
      if (numDim !== dataPoint.vector.length) {
        logging.setModalMessage(
            'Parsing failed. Vector dimensions do not match');
        throw Error('Parsing failed');
      }
      if (numDim <= 1) {
        logging.setModalMessage(
            'Parsing failed. Found a vector with only one dimension?');
        throw Error('Parsing failed');
      }
    });
    return data;
  }, TENSORS_MSG_ID).then(dataPoints => {
    logging.setModalMessage(null, TENSORS_MSG_ID);
    return dataPoints;
  });
}

function analyzeMetadata(columnNames, pointsMetadata: PointMetadata[]):
    ColumnStats[] {
  let columnStats: ColumnStats[] = columnNames.map(name => {
    return {
      name: name,
      isNumeric: true,
      tooManyUniqueValues: false,
      min: Number.POSITIVE_INFINITY,
      max: Number.NEGATIVE_INFINITY
    };
  });
  let mapOfValues = columnNames.map(() => d3.map<number>());
  pointsMetadata.forEach(metadata => {
    columnNames.forEach((name: string, colIndex: number) => {
      let stats = columnStats[colIndex];
      let map = mapOfValues[colIndex];
      let value = metadata[name];

      // Skip missing values.
      if (value == null) {
        return;
      }

      if (!stats.tooManyUniqueValues) {
        if (map.has(value)) {
          map.set(value, map.get(value) + 1);
        } else {
          map.set(value, 1);
        }
        if (map.size() > NUM_COLORS_COLOR_MAP) {
          stats.tooManyUniqueValues = true;
        }
      }
      if (isNaN(value as any)) {
        stats.isNumeric = false;
      } else {
        metadata[name] = +value;
        stats.min = Math.min(stats.min, +value);
        stats.max = Math.max(stats.max, +value);
      }
    });
  });
  columnStats.forEach((stats, colIndex) => {
    let map = mapOfValues[colIndex];
    if (!stats.tooManyUniqueValues) {
      stats.uniqueEntries = map.entries().map(e => {
        return {label: e.key, count: e.value};
      });
    }
  });
  return columnStats;
}

function parseMetadata(content: string): Promise<MetadataInfo> {
  return runAsyncTask('Parsing metadata...', () => {
    let lines = content.split('\n').filter(line => line.trim().length > 0);
    let hasHeader = lines[0].indexOf('\t') >= 0;
    let pointsMetadata: PointMetadata[] = [];
    // If the first row doesn't contain metadata keys, we assume that the values
    // are labels.
    let columnNames = ['label'];
    if (hasHeader) {
      columnNames = lines[0].split('\t');
      lines = lines.slice(1);
    }
    lines.forEach((line: string) => {
      let rowValues = line.split('\t');
      let metadata: PointMetadata = {};
      pointsMetadata.push(metadata);
      columnNames.forEach((name: string, colIndex: number) => {
        let value = rowValues[colIndex];
        // Normalize missing values.
        value = (value === '' ? null : value);
        metadata[name] = value;
      });
    });
    return {
      stats: analyzeMetadata(columnNames, pointsMetadata),
      pointsInfo: pointsMetadata
    } as MetadataInfo;
  }, METADATA_MSG_ID).then(metadata => {
    logging.setModalMessage(null, METADATA_MSG_ID);
    return metadata;
  });
}

function fetchImage(url: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    let image = new Image();
    image.onload = () => resolve(image);
    image.onerror = (err) => reject(err);
    image.src = url;
  });
}

type DemoDataset = {
  fpath: string; metadata_path?: string; metadata?: DatasetMetadata;
  bookmarks_path?: string;
  shape: [number, number];
};

/** Data provider that loads data from a demo folder. */
class DemoDataProvider implements DataProvider {
  /** List of demo datasets for showing the capabilities of the tool. */
  private static DEMO_DATASETS: {[name: string]: DemoDataset} = {
    'Glove Wiki 5K': {
      shape: [5000, 50],
      fpath: 'wiki_5000_50d_tensors.ssv',
      metadata_path: 'wiki_5000_50d_labels.ssv'
    },
    'Glove Wiki 10K': {
      shape: [10000, 100],
      fpath: 'wiki_10000_100d_tensors.ssv',
      metadata_path: 'wiki_10000_100d_labels.ssv'
    },
    'Glove Wiki 40K': {
      shape: [40000, 100],
      fpath: 'wiki_40000_100d_tensors.ssv',
      metadata_path: 'wiki_40000_100d_labels.ssv'
    },
    'SmartReply 5K': {
      shape: [5000, 256],
      fpath: 'smartreply_5000_256d_tensors.tsv',
      metadata_path: 'smartreply_5000_256d_labels.tsv'
    },
    'SmartReply All': {
      shape: [35860, 256],
      fpath: 'smartreply_full_256d_tensors.tsv',
      metadata_path: 'smartreply_full_256d_labels.tsv'
    },
    'Mnist with images 10K': {
      shape: [10000, 784],
      fpath: 'mnist_10k_784d_tensors.tsv',
      metadata_path: 'mnist_10k_784d_labels.tsv',
      metadata: {
        image:
            {sprite_fpath: 'mnist_10k_sprite.png', single_image_dim: [28, 28]}
      },
    },
    'Iris': {
      shape: [150, 4],
      fpath: 'iris_tensors.tsv',
      metadata_path: 'iris_labels.tsv'
    }
  };
  /** Name of the folder where the demo datasets are stored. */
  private static DEMO_FOLDER = 'data';

  retrieveRuns(callback: (runs: string[]) => void): void {
    callback(['Demo']);
  }

  retrieveCheckpointInfo(run: string, callback: (d: CheckpointInfo) => void)
      : void {
    let tensorsInfo: {[name: string]: TensorInfo} = {};
    for (let name in DemoDataProvider.DEMO_DATASETS) {
      if (!DemoDataProvider.DEMO_DATASETS.hasOwnProperty(name)) {
        continue;
      }
      let demoInfo = DemoDataProvider.DEMO_DATASETS[name];
      tensorsInfo[name] = {
        name: name,
        shape: demoInfo.shape,
        metadataFile: demoInfo.metadata_path,
        bookmarksFile: demoInfo.bookmarks_path
      };
    }
    callback({
      tensors: tensorsInfo,
      checkpointFile: 'Demo datasets',
    });
  }

  getDefaultTensor(run: string, callback: (tensorName: string) => void) {
    callback('SmartReply 5K');
  }

  retrieveTensor(run: string, tensorName: string,
      callback: (ds: DataSet) => void) {
    let demoDataSet = DemoDataProvider.DEMO_DATASETS[tensorName];
    let separator = demoDataSet.fpath.substr(-3) === 'tsv' ? '\t' : ' ';
    let url = `${DemoDataProvider.DEMO_FOLDER}/${demoDataSet.fpath}`;
    logging.setModalMessage('Fetching tensors...', TENSORS_MSG_ID);
    d3.text(url, (error: Error, dataString: string) => {
      if (error) {
        console.error(error);
        logging.setModalMessage('Error loading data.');
        return;
      }
      parseTensors(dataString, separator).then(points => {
        callback(new DataSet(points));
      });
    });
  }

  retrieveMetadata(run: string, tensorName: string,
      callback: (r: MetadataInfo) => void) {
    let demoDataSet = DemoDataProvider.DEMO_DATASETS[tensorName];
    let dataSetPromise: Promise<MetadataInfo> = null;
    if (demoDataSet.metadata_path) {
      dataSetPromise = new Promise<MetadataInfo>((resolve, reject) => {
        logging.setModalMessage('Fetching metadata...', METADATA_MSG_ID);
        d3.text(
            `${DemoDataProvider.DEMO_FOLDER}/${demoDataSet.metadata_path}`,
            (err: Error, rawMetadata: string) => {
              if (err) {
                console.error(err);
                reject(err);
                return;
              }
              resolve(parseMetadata(rawMetadata));
            });
      });
    }
    let spriteMsgId = null;
    let spritesPromise: Promise<HTMLImageElement> = null;
    if (demoDataSet.metadata && demoDataSet.metadata.image) {
      let spriteFilePath = demoDataSet.metadata.image.sprite_fpath;
      spriteMsgId = logging.setModalMessage('Fetching sprite image...');
      spritesPromise =
          fetchImage(`${DemoDataProvider.DEMO_FOLDER}/${spriteFilePath}`);
    }

    // Fetch the metadata and the image in parallel.
    Promise.all([dataSetPromise, spritesPromise]).then(values => {
      if (spriteMsgId) {
        logging.setModalMessage(null, spriteMsgId);
      }
      let [metadata, spriteImage] = values;
      metadata.spriteImage = spriteImage;
      metadata.datasetInfo = demoDataSet.metadata;
      callback(metadata);
    });
  }

  getBookmarks(
      run: string, tensorName: string, callback: (r: State[]) => void) {
    callback([]);
  }
}

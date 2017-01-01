Human or Robot - Kaggle Competition
===================================

TL;DR

This repository contains my solution to the Kaggle competition [Facebook Recruiting IV: Human or Robot?](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot). Datasets and a description of the task can be found on the competition website.

## Required Packages

The code is written in Python and depends on the following packages.

*   [NumPy](http://www.numpy.org)
*   [SciPy](http://www.scipy.org)
*   [scikit-learn](http://scikit-learn.org)
*   [pandas](http://pandas.pydata.org)
*   [PyTables](http://www.pytables.org)

The code runs in Python 2.7. Python 3 support is not tested.

## Running the Code

1.  Download the datasets from the official Kaggle website and extract them into the `data` directory. The `data` directory should contain three files, `train.csv`, `test.csv` and `bids.csv`. After executing the following commands, a SQLite database `bids.db` will be built to ease further data analysis. The database contains indexed data in `bids.csv`, and consumes about 1.35 GB of disk space.

    ```bash
    cd data
    make
    ```

2.  Generate feature files and make prediction.

    ```bash
    cd src
    python features.py
    python prediction.py
    ```

    After running the above commands, a large number of intermediate data files and feature files will be generated in the `workspace` directory (about 1.3 GB). Then prediction of the test set will be written to `workspace/submission`.

    I have only tested the code on a MacBook Pro with SSD and 16 GB memory. Some of the feature extraction procedures have a large memory footprint, so the code may not run on computers with smaller amount of memory.

    Feature extraction can take a few hours on a single-core processor, since some of the features, especially `series_crosscorr` (see below), require a large amount of time to generate. It is also possible to create feature sets by calling functions in `features.py` (possibly in parallel), and that was how I built feature sets incrementally as I explored the data during the competition.

    Luckily, feature extraction need only to be performed once. Training and making prediction is relatively fast. And it is convenient to try different classifiers using pre-computed feature files. I found that [random forest](https://en.wikipedia.org/wiki/Random_forest) worked best in this task, and you may refer to `prediction.py` for how I load pre-computed features from files and build a scikit-learn pipeline. You can also do cross-validation and perform a grid search in the parameter space, as is exemplified in the following code snippet.

    ```python
    import logging
    logging.basicConfig(level=logging.INFO)

    import prediction

    # cross validation
    auc = prediction.cross_validation(k=3)

    # parameter grid search
    from sklearn.grid_search import GridSearchCV

    train, label = prediction.get_training_data()
    pipeline = prediction.create_pipeline()
    params = {
        'classifier__max_features': ['log2', 'auto'],
        'classifier__n_estimators': [100, 200, 300],
    }
    gs = GridSearchCV(pipeline, params, scoring='roc_auc')
    gs.fit(train, label)
    for x in gs.grid_scores_:
        print x
    print gs.best_params_
    ```

## Intermediate Data

I have created a few intermediate datasets for analysis purpose. These datasets are also needed for feature extraction described in the next section. All intermediate data are stored in the `workspace` directory.

*   `frequencies/{attribute}.csv` where `{attribute}` can be `bidder_id`, `auction`, `merchandise`, `device`, `country`, `ip`, `url`

    These files stores counts that different values of an attribute have appeared in the bid stream. For example, `frequencies/auction.csv` stores number of bids for all auctions.

*   `graphs/{attribute}.csv.gz` where `{attribute}` can be `auction`, `merchandise`, `device`, `country`, `ip`, `url`

    These files are edge lists of bidder-attribute bipartite graphs in gzipped CSV format. For example, in the `graphs/ip.csv.gz` file, the first two columns are `bidder_id` and `ip`, and the third column, `weight`, indicates how many times a bidder bids using a specific IP address.

*   `cooccurrence/{attribute}.pickle.gz` where `{attribute}` can be `auction`, `merchandise`, `device`, `country`, `ip`, `url`

    These files are edge lists of bidder-bidder cooccurrence graphs in gzipped pickle format. For example, in the `cooccurrence/ip.pickle.gz` file, the first two columns are `bidder_id_x` and `bidder_id_y`, and the third column, `weight`, indicates how many IP addresses that the two bidders have shared in the past.

*   `misc/timestamp_stat.csv`

    This file stores maximum, minimum values for the raw timestamps and minimum gap between timestamps. These values are useful for reverse engineering the original mangled timestamps. I have found that timestamps can be transformed so that the bid stream fits in a one-month duration. The transformation can be used to determine the scale of time series discussed below.

*   `series/bid_count_{rate}.h5` where `{rate}` can be `10s`, `30s`, `1min`, `10min`, `30min`, `1h`, `6h`, `12h`, `1d`, ranging from 10 seconds to 1 day

    These files are bid count series in HDF5 format. For each bidder, I count how many bids are made in each time interval of the resolution specified by `{rate}` (e.g., number of bids every minute).

*   `series/unique_count_{rate}/{attribute}.h5` where `{rate}` is defined the same as above and `{attribute}` can be `auction`, `device`, `country`, `ip`, `url`

    These files are unique attribute count series in HDF5 format. For each bidder, I count the number of unique values of a given attribute in each time interval of different resolutions (e.g., number of unique IP addresses every hour).

## Feature Engineering

In this competition I have adopted a somewhat "brute-force" approach. I extracted a large number of features and let the random forest classifier select the most promising ones.

Feature vectors for each bidder is pre-computed and stored in hierarchy in the `workspace/features` directory. Each feature set is a CSV file containing a set of numeric value features for all bidders. The first column is always `bidder_id`, followed by a comma-separated feature vector. The CSV file contains a header line for feature names. The feature vectors for some bidders may be missing due to their absence from `bids.csv`. It is also possible that some features for some bidders is missing due to their limited bid counts. Before classification these feature sets are loaded into memory and concatenated, forming a long feature vector for each bidder. Missing values are treated as `NaN` and are handled properly by preprocessors in the prediction pipeline.

Here is a brief description of different types of feature sets.

### Simple Statistics Features

*   `per_auction_freq/{attribute}.csv` where `{attribute}` can be `merchandise`, `device`, `country`, `ip`, `url`

    I selected 100 auctions with the largest bid counts and count the unique values for different attributes (e.g., IP address) in the bid record for each bidder in each auction. This gives `100 * 5 = 500` features.

*   `attribute_weight_stats/{attribute}.csv` where `{attribute}` can be `auction`, `device`, `country`, `ip`, `url`

    These feature sets are statistics for attribute values for each bidder. For example, for a bidder `x` in the `attribute_weight_stats/device` feature set, I first collect the devices that bidder `x` has used, and get the frequencies that these devices have appeared in the entire bid stream. Then statistics for these frequency counts are calculated.

    Each feature set contains statistics `count`, `min`, `max`, `mean`, `std`, `kurtosis`, `percentile_{25,50,75}`.

    This gives `9 * 5 = 45` features.

### Graph Spectrum Features

*   `graph_svd/{attribute}.csv` where `{attribute}` can be `auction`, `merchandise`, `device`, `country`, `ip`, `url`

    I construct biadjacency matrices of bidder-attribute bipartite graphs using `graphs/{attribute}.csv.gz` data files mentioned previously. (For example, in the bidder-IP matrix, each row corresponds to a bidder and each column corresponds to an IP address. Element `(x, y)` with value `w` means that bidder with ID `x` has used IP address `y` for `w` times in all bids.) Left singular vectors of such matrices, truncated to keep components corresponding to the 100 largest singular values, are stored as features. (The exception is `merchandise`, which has only 9 unique choices, so we only get 9 singular values.) This gives `100 * 5 + 9 * 1 = 509` features.

*   `cooccurrence_eigen/{attribute}.csv` where `{attribute}` can be `auction`, `merchandise`, `device`, `country`, `ip`, `url`

    I construct adjacency matrices of the bidder-bidder graphs using `cooccurrence/{attribute}.pickle.gz` data files mentioned above. Components of the eigenvectors corresponding to the 100 largest eigenvalues are stored as features. This gives `100 * 6 = 600` features.

### Timestamps Statistics Features

These features are sample statistics for measurements for each bidder based on timestamps.

*   `response_time_stats.csv`

    Statistics of response time for each bidder. The response time refers to the time difference between a bid and the previous bid (possibly by a different bidder) in the same auction.

*   `interarrival_time_stats.csv`

    Statistics of interarrival time for each bidder. The interarrival time refers to the time difference between two adjacent bids by the same bidder in an auction.

*   `interarrival_steps_stats.csv`

    Statistics of interarrival steps for each bidder. I define "interarrival steps" as the number of bids (possibly by other bidders) between two bids from the same bidder in an auction.

*   `bid_amounts_stats.csv`

    Statistics of numbers of consecutive bids for each bidder. Since each bid has a fixed value, the number of consecutive bids can be treated as the amount for a bid.

The statistics are `count`, `max`, `std`, `mean`, `min`, `percentile_{0,10,20,30,40,50,60,70,80,90}`. For `response_time_stats` and `interarrival_time_stats`, the percentiles are normalized (divided by the maximum).

The above four feature sets give `15 * 4 = 60` features.

### Time Series Features

*   `unique_count_series_stats_{rate}/{attribute}.csv`

    Statistics of the `series/unique_count_{rate}/{attribute}.h5` time series mentioned previously.

    The statistics are `min`, `max`, `mean`, `std`, `kurtosis`, `entropy`, `autocorr_{1,2,3,4,5,6,7,8,9,10}`, `dftpeak_{0,1,2,3,4,5,6,7,8,9}`, `dftquantile_{0,25,50,75,100}`. `autocorr_{t}` is the [auto-correlation](https://en.wikipedia.org/wiki/Autocorrelation) between time series `x[s]` and `x[s+t]`. I apply discrete Fourier transform (DFT) to the time series. `dftpeak_{k}` is the frequency that has the `k`-th largest amplitude. `dftquantile_{q}` is the `q`-quantile of the amplitude of all frequencies.

    When `{rate}` is `12h`, `dftpeak_9` is missing. When `{rate}` is `1d`, `autocorr_{9,10}` and `dftpeak_{4,5,6,7,8,9}` is missing, and `autocorr_{6,7,8}` contain `NaN` only and are thus dropped by `sklearn.preprocessing.Imputer`.

    This gives `9 * 5 * 31 - 5 * 1 - 5 * 11 = 1335` features.

*   `bid_count_series_stats_{rate}.csv`

    Statistics of the `series/bid_count_{rate}.h5` time series mentioned previously.

    This gives `9 * 31 - 1 - 11 = 267` features.

*   `series_crosscorr_{rate}.csv`

    Each feature is named as `{x}_vs_{y}_{t}`, denoting the [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) between time series `x[s]` and `y[s+t]` where `{t}` is the shift between the two time series. I pick `{t}` to be `0`, `1`, `2`, and `{x}` and `{y}` can be chosen from `unique_auction`, `unique_device`, `unique_country`, `unique_ip`, `unique_url`, `bid`, corresponding to the 6 types of time series (`series/bid_count_{rate}.h5` and `series/unique_count_{rate}/{attribute}.h5` with the same rate) introduced above.

    This gives `9 * 6 * (6 - 1) * 3 = 810` features.

In summary, there are `500 + 45 + 509 + 600 + 60 + 1335 + 267 + 810 = 4126` features in total.

## Results

The two final submissions I have made were from two runs of the random forest classifier using the features described above. The ROC AUC scores were `0.93920` and `0.93778` on the private leaderboard. (The scores were `0.91776` and `0.90906` on the public leaderboard, respectively.) I ranked the **10th** among the 985 teams.

The function `prediction.get_feature_importance()` produces a sorted list of features along with their importance scores after training a classifier. Although the results vary across runs, one can see that the most useful features are usually auto-correlation, cross-correlation, DFT quantile and graph SVD features.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::pair;


class DisjointSetUnion {
private:
    vector<int> parent_;
    vector<int> ranks_;

public:
    explicit DisjointSetUnion(size_t size)
            : parent_()
            , ranks_(size, 0) {
        parent_.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            parent_.push_back(i);
        }
    }

    int find(int node) {
        if (parent_[node] != node) {
            parent_[node] = find(parent_[node]);
        }
        return parent_[node];
    }

    void union_sets(int first, int second) {
        int first_root = find(first);
        int second_root = find(second);
        if (first_root == second_root) {
            return;
        }

        if (ranks_[first_root] < ranks_[second_root]) {
            parent_[first_root] = second_root;
        } else if (ranks_[first_root] > ranks_[second_root]) {
            parent_[second_root] = first_root;
        } else {
            parent_[second_root] = first_root;
            ++ranks_[first_root];
        }
    }
};


struct Edge {
    size_t from;
    size_t to;
    double weight;
};


// Map arbitrary labels to 0, ..., (n-1) labels
vector<size_t> RenumerateLabels(const vector<size_t>& rawLabels) {
    vector<int> rawToNew(rawLabels.size(), -1);
    size_t indexesUsed = 0;
    vector<size_t> newLabels(rawLabels.size());
    for (size_t i = 0; i < rawLabels.size(); ++i) {
        size_t oldLabel = rawLabels[i];
        if (rawToNew[oldLabel] == -1) {
            rawToNew[oldLabel] = indexesUsed;
            ++indexesUsed;
        }
        newLabels[i] = rawToNew[oldLabel];
    }
    return newLabels;
}


vector<size_t> ClusterGraphMST(vector<Edge> edges,  // copy for sorting
                               size_t vertexCount,
                               size_t clusterCount) {
    size_t edgesMSTCount = vertexCount - clusterCount;
    DisjointSetUnion setUnion(vertexCount);
    // uncomment the following in case of need to store edges of MST
    // vector<Edge> MST(edgesMSTCount);
    size_t edgesAddedCount = 0;
    size_t indexOfEdge = 0;
    auto comp = [&](const Edge& lhs, const Edge& rhs) { return lhs.weight < rhs.weight; };
    std::sort(edges.begin(), edges.end(), comp);
    while (edgesAddedCount < edgesMSTCount) {
        auto consideringEdge = edges[indexOfEdge++];
        int firstParent = setUnion.find(consideringEdge.from);
        int secondParent = setUnion.find(consideringEdge.to);
        if (firstParent != secondParent) {
            // MST[edgesAddedCount++] = consideringEdge;
            ++edgesAddedCount;
            setUnion.union_sets(consideringEdge.from, consideringEdge.to);
        }
    }

    vector<size_t> rawLabels(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        rawLabels[i] = setUnion.find(i);
    }
    vector<size_t> cleanLabels = RenumerateLabels(rawLabels);
    return cleanLabels;
}

template <typename T, typename Dist>
vector<Edge> PairwiseDistances(vector<T> objects, Dist distance) {
    vector<Edge> edges;
    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {
            edges.push_back({i, j, distance(objects[i], objects[j])});
        }
    }
    return edges;
}


template <typename T, typename Dist>
vector<size_t> ClusterMST(const vector<T>& objects, Dist distance, size_t clusterCount) {
    vector<Edge> edges = PairwiseDistances(objects, distance);
    return ClusterGraphMST(edges, objects.size(), clusterCount);
}

// using the Fisher-Yates shuffle
template <typename Iter>
Iter random_unique(Iter begin, Iter end, size_t num_random) {
    long left = std::distance(begin, end);
    std::srand(std::time(0));
    while (num_random--) {
        Iter r = begin;
        std::advance(r, std::rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

template <typename T>
vector<T> RandomSubset(vector<T> objects,  // copy for shuffling
                       size_t subsetSize) {
    random_unique(objects.begin(), objects.end(), subsetSize);
    vector<T> subset(subsetSize);
    for (size_t i = 0; i < subsetSize; ++i) {
        subset[i] = objects[i];
    }
    return subset;
}

template<typename T, typename Dist>
size_t NearestMean(const T& object, Dist distance, const vector<T>& means) {
    double minDistance = distance(object, means[0]);
    size_t indexOfNearestMean = 0;
    for (size_t i = 1; i < means.size(); ++i) {
        if (distance(object, means[i]) < minDistance) {
            minDistance = distance(object, means[i]);
            indexOfNearestMean = i;
        }
    }
    return indexOfNearestMean;
};

template <typename T, typename Dist>
pair<vector<size_t>, vector<T>> ClusterKMeans(const vector<T>& objects,
                                              Dist distance,
                                              size_t clusterCount) {
    vector<T> means = RandomSubset(objects, clusterCount);
    const int ATTRIBUTES_NUMBER = 2;
    vector<size_t> labels(objects.size(), 0);
    while (true) {
        vector<vector<double>> averageCoordinate(clusterCount,
                vector<double>(ATTRIBUTES_NUMBER, 0.));
        vector<int> objectsInClusterCount(clusterCount);
        for (size_t i = 0; i < objects.size(); ++i) {
            size_t newLabel = NearestMean(objects[i], distance, means);
            labels[i] = newLabel;
            vector<double> attributes = objects[i].getAttributes();
            for (int i = 0; i < ATTRIBUTES_NUMBER; ++i) {
                averageCoordinate[newLabel][i] += attributes[i];
            }
            ++objectsInClusterCount[newLabel];
        }
        vector<T> newMeans(clusterCount);
        for (size_t i = 0; i < clusterCount; ++i) {
            newMeans[i] = {averageCoordinate[i][0] / objectsInClusterCount[i],
                           averageCoordinate[i][1] / objectsInClusterCount[i]};
        }
        if (newMeans == means) {
            break;
        } else {
            means = newMeans;
        }
    }
    return {labels, means};
}

struct Point2D {
    double x, y;

    vector<double> getAttributes() const {
        return {x, y};
    }
};

bool operator ==(const Point2D& lhs, const Point2D& rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

double EuclidianDistance(const Point2D& first, const Point2D& second) {
    return std::sqrt((first.x - second.x) * (first.x - second.x) +
                     (first.y - second.y) * (first.y - second.y));
}

pair<vector<Point2D>, vector<size_t>> Random2DClusters(const vector<Point2D>& centers,
                                 const vector<double>& xVariances,
                                 const vector<double>& yVariances,
                                 size_t pointsCount) {
    auto baseGenerator = std::default_random_engine();
    auto generateCluster = std::uniform_int_distribution<size_t>(0, centers.size() - 1);
    auto generateDeviation = std::normal_distribution<double>();

    vector<Point2D> results;
    vector<size_t> initialLabels(pointsCount);
    for (size_t i = 0; i < pointsCount; ++i) {
        size_t c = generateCluster(baseGenerator);
        double x = centers[c].x + generateDeviation(baseGenerator) * xVariances[c];
        double y = centers[c].y + generateDeviation(baseGenerator) * yVariances[c];
        results.push_back({x, y});
        initialLabels[i] = c;
    }
    return {results, initialLabels};
}


// Generate files for plotting in gnuplot
void GNUPlotClusters2D(const vector<Point2D>& points,
                       const vector<size_t>& labels,
                       size_t clustersCount,
                       const string& outFolder) {
    std::ofstream scriptOut(outFolder + "/script.txt");
    scriptOut << "plot ";

    for (size_t cluster = 0; cluster < clustersCount; ++cluster) {
        string filename = std::to_string(cluster) + ".dat";
        std::ofstream fileOut(outFolder + "/" + filename);
        scriptOut << "\"" << filename << "\"" << " with points, ";

        for (size_t i = 0; i < points.size(); ++i) {
            if (labels[i] == cluster) {
                fileOut << points[i].x << "\t" << points[i].y << "\n";
            }
        }
    }
}

// Generate files for plotting in gnuplot with marking centers (means) of clusters
void GNUPlotClusters2D(const vector<Point2D>& points,
                       const vector<size_t>& labels,
                       const vector<Point2D>& means,
                       size_t clustersCount,
                       const string& outFolder) {
    std::ofstream scriptOut(outFolder + "/script.txt");
    scriptOut << "plot ";

    for (size_t cluster = 0; cluster < clustersCount; ++cluster) {
        string filename = std::to_string(cluster) + ".dat";
        std::ofstream fileOut(outFolder + "/" + filename);
        scriptOut << "\"" << filename << "\"" << " with points, ";

        for (size_t i = 0; i < points.size(); ++i) {
            if (labels[i] == cluster) {
                fileOut << points[i].x << "\t" << points[i].y << "\n";
            }
        }
    }
    string filename = "means.dat";
    std::ofstream fileOut(outFolder + "/" + filename);
    for (size_t i = 0; i < means.size(); ++i) {
        fileOut << means[i].x << "\t" << means[i].y << "\n";
    }
    scriptOut << "\"" << filename << "\"" << " with points pt 5 ps 2 lc 8, ";
}


int main() {
    vector<Point2D> points;
    vector<size_t> initial_labels;
    const size_t CLUSTER_COUNT = 3;
    tie(points, initial_labels) = Random2DClusters(
            {{1, 1}, {2, 1}, {1.5, 1.5}},
            {0.1, 0.3, 0.2},
            {0.3, 0.1, 0.2},
            4321);

    // store .dat files in /plot_initial folder so as to see correct clusters
    GNUPlotClusters2D(points, initial_labels, CLUSTER_COUNT, "./plot_initial");

    // plot_base folder stores raw data of generated points
    // (assuming the whole data as one cluster)
    vector<size_t> labels(points.size(), 0);
    GNUPlotClusters2D(points, labels, 1, "./plot_base");

    // plot_mst folder stores clusters computed by mst algorithm
    labels = ClusterMST(points, EuclidianDistance, CLUSTER_COUNT);
    GNUPlotClusters2D(points, labels, CLUSTER_COUNT, "./plot_mst");

    // plot_kmeans folder stores clusters computed by K-Means algorithm
    vector<Point2D> means;
    std::tie(labels, means) = ClusterKMeans(points, EuclidianDistance, CLUSTER_COUNT);
    GNUPlotClusters2D(points, labels, means, CLUSTER_COUNT, "./plot_kmeans");

    return 0;
}


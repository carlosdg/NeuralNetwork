var Neuron = (function () {
    // We need to distinguish between two types of neurons, the normal ones and
    // the ones with a constant output (input neurons as well as auxiliar neurons
    // with the constant 1 as output)
    function Neuron(numberOfInputs, isConstantOutputNeuron_, outputVal_) {
        if (outputVal_ === void 0) { outputVal_ = 1; }
        this.isConstantOutputNeuron_ = isConstantOutputNeuron_;
        this.outputVal_ = outputVal_;
        // Weights needed to compute the output value of the neuron
        // In a neural network draw, these weights are the ones that comes to
        // this neuron
        this.weights_ = [];
        this.derivedOutputVal_ = 1;
        this.delta_ = 1;
        // The network has to add an aditional neuron in each layer with a constant
        // 1 as output to have the bias as a weight.
        for (var i = 0; i < numberOfInputs; ++i) {
            this.weights_.push(Math.random());
        }
    }
    // Getter
    Neuron.prototype.getOutput = function () { return this.outputVal_; };
    // Output setter, only for the special neurons.
    // This is so we can change the input of the neural network
    Neuron.prototype.setOutput = function (newVal) {
        if (this.isConstantOutputNeuron_) {
            this.outputVal_ = newVal;
        }
        else {
            throw Error("Trying to change the output manually of a \"normal\" neuron");
        }
    };
    // Activation function, sigmoid in this case.
    Neuron.activationFunction = function (input) {
        return (1 / (1 + Math.exp(-input)));
    };
    Neuron.rateOfChangeOfActivationFunction = function (activatedInput) {
        return ((1 - activatedInput) * activatedInput);
    };
    // @param inputFeatures: vector of inputs
    // @return: This object is return so we can chain methods.
    Neuron.prototype.feed = function (inputFeatures) {
        // If for some reason feed() is called with an input neuron or an
        // extra neuron we don't have to do anything
        if (this.isConstantOutputNeuron_)
            return this;
        // Checking that the dot product can be done.
        var length = inputFeatures.length;
        if (length !== this.weights_.length) {
            throw Error("Cannot perform dot product. Vector sizes don't match");
        }
        // Weighted sum
        this.outputVal_ = 0;
        for (var i = 0; i < length; ++i) {
            this.outputVal_ += inputFeatures[i].outputVal_ * this.weights_[i];
        }
        // Applying activation function
        this.outputVal_ = Neuron.activationFunction(this.outputVal_);
        // Storing derivative
        this.derivedOutputVal_ = Neuron.rateOfChangeOfActivationFunction(this.outputVal_);
        return this;
    };
    // Delta calculation functions. See README for the math behind this.
    Neuron.prototype.calculateOutputLayerDelta = function (yReal) {
        // DerivativeOf( a[k][q] )WithRespectTo( s[k][q] ) * ( a[k][q] - yReal[q] )
        this.delta_ = this.derivedOutputVal_ * (this.outputVal_ - yReal);
        return this;
    };
    Neuron.prototype.calculateHiddenLayerDelta = function (nextLayerNeurons, q, iAmLastHiddenLayer) {
        if (iAmLastHiddenLayer === void 0) { iAmLastHiddenLayer = false; }
        var sum = 0, nextLayerSize = nextLayerNeurons.length;
        if (!iAmLastHiddenLayer) {
            // If this node is in the last layer, the next layer will be the output layer.
            // The output layer doesn't have extra neurons, but the hidden layers do, so
            // We have to take that into account
            nextLayerSize -= 1;
        }
        for (var i = 0; i < nextLayerSize; ++i) {
            // W[k][q][i] * Delta[k+1][i]
            sum += nextLayerNeurons[i].weights_[q] * nextLayerNeurons[i].delta_;
            // Note that because of the way we store the weights, in the code it seems like we
            // are multiplying the weights and the deltas of the same layer, but remember that
            // in the neurons we store the weights that they has as input, those would be
            // the weights of the k-1 layer.
        }
        this.delta_ = this.derivedOutputVal_ * sum;
        return this;
    };
    // Update weight function
    Neuron.prototype.updateWeights = function (learningRate, prevLayerNeurons) {
        // Gradient descent:
        // W[k][p][q] <-- W[k][p][q] - Alpha * DerivativeOf( Error )WithRespectTo( W[k][p][q] )
        // For all weights
        // Same formula with delta
        // W[k][p][q] <-- W[k][p][q] - Alpha * a[k][p] * Delta[k+1][q]
        // For all weights
        for (var i = 0, length_1 = this.weights_.length; i < length_1; ++i) {
            this.weights_[i] -= learningRate * (prevLayerNeurons[i].outputVal_ * this.delta_);
        }
        return this;
    };
    return Neuron;
}());
var NeuralNetwork = (function () {
    function NeuralNetwork(topology, learningRate, numIterationsGD) {
        if (learningRate === void 0) { learningRate = 0.01; }
        if (numIterationsGD === void 0) { numIterationsGD = 1000; }
        this.learningRate = learningRate;
        this.numIterationsGD = numIterationsGD;
        this.inputLayer = [];
        this.hiddenLayers = [];
        this.outputLayer = [];
        // Input layer
        for (var i = topology.numInputNeurons; i > 0; --i) {
            this.inputLayer.push(new Neuron(0, true));
        }
        // Add the extra neuron with a constant 1 as output
        this.inputLayer.push(new Neuron(0, true, 1));
        // Hidden layers
        if (topology.numHiddenLayerNeurons.length === 0) {
            throw Error("This implementation requires at least one hidden layer");
        }
        var numNeuronsPrevLayer = this.inputLayer.length;
        // Create the hidden layers
        for (var i = 0, length_2 = topology.numHiddenLayerNeurons.length; i < length_2; ++i) {
            this.hiddenLayers.push([]);
            // Construct the neurons in each hidden layer
            for (var j = topology.numHiddenLayerNeurons[i]; j > 0; --j) {
                this.hiddenLayers[i].push(new Neuron(numNeuronsPrevLayer, false));
            }
            // Add the extra neuron
            this.hiddenLayers[i].push(new Neuron(0, true, 1));
            // Update the number of neurons in the previous layer variable for the next layer
            numNeuronsPrevLayer = this.hiddenLayers[i].length;
        }
        // Output layer
        for (var i = topology.numOutputLayerNeurons; i > 0; --i) {
            this.outputLayer.push(new Neuron(numNeuronsPrevLayer, false));
        }
    }
    NeuralNetwork.prototype.predict = function (inputFeatures) {
        // Check that length of input is correct
        var numInputNeurons = this.inputLayer.length;
        if (inputFeatures.length !== numInputNeurons - 1) {
            throw Error("Input is of size " + inputFeatures.length + ". Expected size " + (numInputNeurons - 1));
        }
        // Set input neurons values. Very important to not include the extra neuron
        for (var i = 0; i < numInputNeurons - 1; ++i) {
            this.inputLayer[i].setOutput(inputFeatures[i]);
        }
        // Feed foward. Hidden layers
        // First hidden layer. There is no need to call feed on the extra neuron
        for (var i = 0, numNeurons = this.hiddenLayers[0].length - 1; i < numNeurons; ++i) {
            this.hiddenLayers[0][i].feed(this.inputLayer);
        }
        // Rest of hidden layers
        for (var i = 1, numHiddenLayers = this.hiddenLayers.length; i < numHiddenLayers; ++i) {
            for (var j = 0, numNeurons = this.hiddenLayers[i].length - 1; j < numNeurons; ++j) {
                this.hiddenLayers[i][j].feed(this.hiddenLayers[i - 1]);
            }
        }
        // Feed foward. Output neurons
        var result = []; // Variable with the results to return
        for (var i = 0, numNeurons = this.outputLayer.length; i < numNeurons; ++i) {
            var lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
            // Feed the neuron and push the output to results
            result.push(this.outputLayer[i]
                .feed(lastHiddenLayer)
                .getOutput());
        }
        return result;
    };
    NeuralNetwork.prototype.train = function (X, Y) {
        // Brief pseudocode of this part
        // For each iteration of Gradient Descent do:
        // For each input do:
        // Make a prediction
        // Calculate deltas (from left to right, very important the order)
        // Update weights
        // For each iteration of GD
        for (var gdIteration = this.numIterationsGD; gdIteration > 0; --gdIteration) {
            // For each input
            for (var i = 0, numInputs = X.length; i < numInputs; ++i) {
                // Make a prediction (output neurons store it and use it in the delta calculation)
                this.predict(X[i]);
                // Calculate output layer deltas
                for (var j = 0, numOutputLayerNeurons = this.outputLayer.length; j < numOutputLayerNeurons; ++j) {
                    this.outputLayer[j].calculateOutputLayerDelta(Y[i][j]);
                }
                // Calculate hidden layer deltas
                // First the last hidden layer, which we have to pass the output layer for delta calculation
                // Note that we have to calculate deltas of extra neurons, otherwise we could not update the
                // bias weights. However, when calculating the deltas we don't consider the extra neuron of the
                // next layer (they are not part of the real network after all, they are just there for convenience, so
                // we can consider the bias as one more weight)
                for (var j = 0, lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1], numNeurons = lastHiddenLayer.length; j < numNeurons; ++j) {
                    lastHiddenLayer[j].calculateHiddenLayerDelta(this.outputLayer, j, true);
                }
                // Now the rest of the hidden layers
                for (var j = this.hiddenLayers.length - 2; j >= 0; --j) {
                    for (var n = 0, numNeurons = this.hiddenLayers[j].length; n < numNeurons; ++n) {
                        this.hiddenLayers[j][n].calculateHiddenLayerDelta(this.hiddenLayers[j + 1], n);
                    }
                }
                // Update hidden layer weights
                // First hidden layer
                for (var j = 0, numNeurons = this.hiddenLayers[0].length - 1; j < numNeurons; ++j) {
                    this.hiddenLayers[0][j].updateWeights(this.learningRate, this.inputLayer);
                }
                // Rest of hidden layers
                for (var j = 1, numHiddenLayers = this.hiddenLayers.length; j < numHiddenLayers; ++j) {
                    for (var n = 0, numNeurons = this.hiddenLayers[j].length - 1; n < numNeurons; ++n) {
                        this.hiddenLayers[j][n].updateWeights(this.learningRate, this.hiddenLayers[j - 1]);
                    }
                }
                // Update output layer weights
                for (var n = 0, numNeurons = this.outputLayer.length; n < numNeurons; ++n) {
                    var lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
                    this.outputLayer[n].updateWeights(this.learningRate, lastHiddenLayer);
                }
            }
        }
        return this;
    };
    return NeuralNetwork;
}());
// Training data
var X = [], Y = [];
// Iris dataset
var irisDataSet = [
    [[5.1, 3.5, 1.4, 0.2], [1, 0, 0]],
    [[4.9, 3.0, 1.4, 0.2], [1, 0, 0]],
    [[4.7, 3.2, 1.3, 0.2], [1, 0, 0]],
    [[4.6, 3.1, 1.5, 0.2], [1, 0, 0]],
    [[5.0, 3.6, 1.4, 0.2], [1, 0, 0]],
    [[5.4, 3.9, 1.7, 0.4], [1, 0, 0]],
    [[4.6, 3.4, 1.4, 0.3], [1, 0, 0]],
    [[5.0, 3.4, 1.5, 0.2], [1, 0, 0]],
    [[4.4, 2.9, 1.4, 0.2], [1, 0, 0]],
    [[4.9, 3.1, 1.5, 0.1], [1, 0, 0]],
    [[5.4, 3.7, 1.5, 0.2], [1, 0, 0]],
    [[4.8, 3.4, 1.6, 0.2], [1, 0, 0]],
    [[4.8, 3.0, 1.4, 0.1], [1, 0, 0]],
    [[4.3, 3.0, 1.1, 0.1], [1, 0, 0]],
    [[5.8, 4.0, 1.2, 0.2], [1, 0, 0]],
    [[5.7, 4.4, 1.5, 0.4], [1, 0, 0]],
    [[5.4, 3.9, 1.3, 0.4], [1, 0, 0]],
    [[5.1, 3.5, 1.4, 0.3], [1, 0, 0]],
    [[5.7, 3.8, 1.7, 0.3], [1, 0, 0]],
    [[5.1, 3.8, 1.5, 0.3], [1, 0, 0]],
    [[5.4, 3.4, 1.7, 0.2], [1, 0, 0]],
    [[5.1, 3.7, 1.5, 0.4], [1, 0, 0]],
    [[4.6, 3.6, 1.0, 0.2], [1, 0, 0]],
    [[5.1, 3.3, 1.7, 0.5], [1, 0, 0]],
    [[4.8, 3.4, 1.9, 0.2], [1, 0, 0]],
    [[5.0, 3.0, 1.6, 0.2], [1, 0, 0]],
    [[5.0, 3.4, 1.6, 0.4], [1, 0, 0]],
    [[5.2, 3.5, 1.5, 0.2], [1, 0, 0]],
    [[5.2, 3.4, 1.4, 0.2], [1, 0, 0]],
    [[4.7, 3.2, 1.6, 0.2], [1, 0, 0]],
    [[4.8, 3.1, 1.6, 0.2], [1, 0, 0]],
    [[5.4, 3.4, 1.5, 0.4], [1, 0, 0]],
    [[5.2, 4.1, 1.5, 0.1], [1, 0, 0]],
    [[5.5, 4.2, 1.4, 0.2], [1, 0, 0]],
    [[4.9, 3.1, 1.5, 0.1], [1, 0, 0]],
    [[5.0, 3.2, 1.2, 0.2], [1, 0, 0]],
    [[5.5, 3.5, 1.3, 0.2], [1, 0, 0]],
    [[4.9, 3.1, 1.5, 0.1], [1, 0, 0]],
    [[4.4, 3.0, 1.3, 0.2], [1, 0, 0]],
    [[5.1, 3.4, 1.5, 0.2], [1, 0, 0]],
    [[5.0, 3.5, 1.3, 0.3], [1, 0, 0]],
    [[4.5, 2.3, 1.3, 0.3], [1, 0, 0]],
    [[4.4, 3.2, 1.3, 0.2], [1, 0, 0]],
    [[5.0, 3.5, 1.6, 0.6], [1, 0, 0]],
    [[5.1, 3.8, 1.9, 0.4], [1, 0, 0]],
    [[4.8, 3.0, 1.4, 0.3], [1, 0, 0]],
    [[5.1, 3.8, 1.6, 0.2], [1, 0, 0]],
    [[4.6, 3.2, 1.4, 0.2], [1, 0, 0]],
    [[5.3, 3.7, 1.5, 0.2], [1, 0, 0]],
    [[5.0, 3.3, 1.4, 0.2], [1, 0, 0]],
    [[7.0, 3.2, 4.7, 1.4], [0, 1, 0]],
    [[6.4, 3.2, 4.5, 1.5], [0, 1, 0]],
    [[6.9, 3.1, 4.9, 1.5], [0, 1, 0]],
    [[5.5, 2.3, 4.0, 1.3], [0, 1, 0]],
    [[6.5, 2.8, 4.6, 1.5], [0, 1, 0]],
    [[5.7, 2.8, 4.5, 1.3], [0, 1, 0]],
    [[6.3, 3.3, 4.7, 1.6], [0, 1, 0]],
    [[4.9, 2.4, 3.3, 1.0], [0, 1, 0]],
    [[6.6, 2.9, 4.6, 1.3], [0, 1, 0]],
    [[5.2, 2.7, 3.9, 1.4], [0, 1, 0]],
    [[5.0, 2.0, 3.5, 1.0], [0, 1, 0]],
    [[5.9, 3.0, 4.2, 1.5], [0, 1, 0]],
    [[6.0, 2.2, 4.0, 1.0], [0, 1, 0]],
    [[6.1, 2.9, 4.7, 1.4], [0, 1, 0]],
    [[5.6, 2.9, 3.6, 1.3], [0, 1, 0]],
    [[6.7, 3.1, 4.4, 1.4], [0, 1, 0]],
    [[5.6, 3.0, 4.5, 1.5], [0, 1, 0]],
    [[5.8, 2.7, 4.1, 1.0], [0, 1, 0]],
    [[6.2, 2.2, 4.5, 1.5], [0, 1, 0]],
    [[5.6, 2.5, 3.9, 1.1], [0, 1, 0]],
    [[5.9, 3.2, 4.8, 1.8], [0, 1, 0]],
    [[6.1, 2.8, 4.0, 1.3], [0, 1, 0]],
    [[6.3, 2.5, 4.9, 1.5], [0, 1, 0]],
    [[6.1, 2.8, 4.7, 1.2], [0, 1, 0]],
    [[6.4, 2.9, 4.3, 1.3], [0, 1, 0]],
    [[6.6, 3.0, 4.4, 1.4], [0, 1, 0]],
    [[6.8, 2.8, 4.8, 1.4], [0, 1, 0]],
    [[6.7, 3.0, 5.0, 1.7], [0, 1, 0]],
    [[6.0, 2.9, 4.5, 1.5], [0, 1, 0]],
    [[5.7, 2.6, 3.5, 1.0], [0, 1, 0]],
    [[5.5, 2.4, 3.8, 1.1], [0, 1, 0]],
    [[5.5, 2.4, 3.7, 1.0], [0, 1, 0]],
    [[5.8, 2.7, 3.9, 1.2], [0, 1, 0]],
    [[6.0, 2.7, 5.1, 1.6], [0, 1, 0]],
    [[5.4, 3.0, 4.5, 1.5], [0, 1, 0]],
    [[6.0, 3.4, 4.5, 1.6], [0, 1, 0]],
    [[6.7, 3.1, 4.7, 1.5], [0, 1, 0]],
    [[6.3, 2.3, 4.4, 1.3], [0, 1, 0]],
    [[5.6, 3.0, 4.1, 1.3], [0, 1, 0]],
    [[5.5, 2.5, 4.0, 1.3], [0, 1, 0]],
    [[5.5, 2.6, 4.4, 1.2], [0, 1, 0]],
    [[6.1, 3.0, 4.6, 1.4], [0, 1, 0]],
    [[5.8, 2.6, 4.0, 1.2], [0, 1, 0]],
    [[5.0, 2.3, 3.3, 1.0], [0, 1, 0]],
    [[5.6, 2.7, 4.2, 1.3], [0, 1, 0]],
    [[5.7, 3.0, 4.2, 1.2], [0, 1, 0]],
    [[5.7, 2.9, 4.2, 1.3], [0, 1, 0]],
    [[6.2, 2.9, 4.3, 1.3], [0, 1, 0]],
    [[5.1, 2.5, 3.0, 1.1], [0, 1, 0]],
    [[5.7, 2.8, 4.1, 1.3], [0, 1, 0]],
    [[6.3, 3.3, 6.0, 2.5], [0, 0, 1]],
    [[5.8, 2.7, 5.1, 1.9], [0, 0, 1]],
    [[7.1, 3.0, 5.9, 2.1], [0, 0, 1]],
    [[6.3, 2.9, 5.6, 1.8], [0, 0, 1]],
    [[6.5, 3.0, 5.8, 2.2], [0, 0, 1]],
    [[7.6, 3.0, 6.6, 2.1], [0, 0, 1]],
    [[4.9, 2.5, 4.5, 1.7], [0, 0, 1]],
    [[7.3, 2.9, 6.3, 1.8], [0, 0, 1]],
    [[6.7, 2.5, 5.8, 1.8], [0, 0, 1]],
    [[7.2, 3.6, 6.1, 2.5], [0, 0, 1]],
    [[6.5, 3.2, 5.1, 2.0], [0, 0, 1]],
    [[6.4, 2.7, 5.3, 1.9], [0, 0, 1]],
    [[6.8, 3.0, 5.5, 2.1], [0, 0, 1]],
    [[5.7, 2.5, 5.0, 2.0], [0, 0, 1]],
    [[5.8, 2.8, 5.1, 2.4], [0, 0, 1]],
    [[6.4, 3.2, 5.3, 2.3], [0, 0, 1]],
    [[6.5, 3.0, 5.5, 1.8], [0, 0, 1]],
    [[7.7, 3.8, 6.7, 2.2], [0, 0, 1]],
    [[7.7, 2.6, 6.9, 2.3], [0, 0, 1]],
    [[6.0, 2.2, 5.0, 1.5], [0, 0, 1]],
    [[6.9, 3.2, 5.7, 2.3], [0, 0, 1]],
    [[5.6, 2.8, 4.9, 2.0], [0, 0, 1]],
    [[7.7, 2.8, 6.7, 2.0], [0, 0, 1]],
    [[6.3, 2.7, 4.9, 1.8], [0, 0, 1]],
    [[6.7, 3.3, 5.7, 2.1], [0, 0, 1]],
    [[7.2, 3.2, 6.0, 1.8], [0, 0, 1]],
    [[6.2, 2.8, 4.8, 1.8], [0, 0, 1]],
    [[6.1, 3.0, 4.9, 1.8], [0, 0, 1]],
    [[6.4, 2.8, 5.6, 2.1], [0, 0, 1]],
    [[7.2, 3.0, 5.8, 1.6], [0, 0, 1]],
    [[7.4, 2.8, 6.1, 1.9], [0, 0, 1]],
    [[7.9, 3.8, 6.4, 2.0], [0, 0, 1]],
    [[6.4, 2.8, 5.6, 2.2], [0, 0, 1]],
    [[6.3, 2.8, 5.1, 1.5], [0, 0, 1]],
    [[6.1, 2.6, 5.6, 1.4], [0, 0, 1]],
    [[7.7, 3.0, 6.1, 2.3], [0, 0, 1]],
    [[6.3, 3.4, 5.6, 2.4], [0, 0, 1]],
    [[6.4, 3.1, 5.5, 1.8], [0, 0, 1]],
    [[6.0, 3.0, 4.8, 1.8], [0, 0, 1]],
    [[6.9, 3.1, 5.4, 2.1], [0, 0, 1]],
    [[6.7, 3.1, 5.6, 2.4], [0, 0, 1]],
    [[6.9, 3.1, 5.1, 2.3], [0, 0, 1]],
    [[5.8, 2.7, 5.1, 1.9], [0, 0, 1]],
    [[6.8, 3.2, 5.9, 2.3], [0, 0, 1]],
    [[6.7, 3.3, 5.7, 2.5], [0, 0, 1]],
    [[6.7, 3.0, 5.2, 2.3], [0, 0, 1]],
    [[6.3, 2.5, 5.0, 1.9], [0, 0, 1]],
    [[6.5, 3.0, 5.2, 2.0], [0, 0, 1]],
    [[6.2, 3.4, 5.4, 2.3], [0, 0, 1]],
    [[5.9, 3.0, 5.1, 1.8], [0, 0, 1]]
];
// Not using the entire dataset, half will be for testing (that's why i+=2)
for (var i = 0, numData = irisDataSet.length; i < numData; i += 2) {
    X.push(irisDataSet[i][0]);
    Y.push(irisDataSet[i][1]);
}
// Training set of logic functions
// for (let i = 0; i < 100; ++i){
//     X.push([ 0, 0 ]);  Y.push([ 0 ]);
//     X.push([ 0, 1 ]);  Y.push([ 1 ]);
//     X.push([ 1, 0 ]);  Y.push([ 1 ]);
//     X.push([ 1, 1 ]);  Y.push([ 0 ]);
// }
console.log(X, Y);
// Construct and train network
var net = new NeuralNetwork({
    numInputNeurons: 4,
    numHiddenLayerNeurons: [10, 5],
    numOutputLayerNeurons: 3
}, 0.01, 100000);
net.train(X, Y);
console.log(net);
// Show the prediction on the whole data set as well as the labels.
for (var i = 0; i < irisDataSet.length; ++i) {
    console.log(net.predict(irisDataSet[i][0]));
    console.log(irisDataSet[i][1] + '\n\n\n');
}

#pragma once
#include <vector>
#include <unordered_map>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>

class sample {
	std::vector<uint8_t> values;
	std::string key;
public:
	sample();
	sample(std::string arg_key);

	// Get a pointer to a value-vector of this sample
	std::vector<uint8_t>* const get_pvalues();
	// Get size of a values vector (size of sample)
	int get_size();
	// Get key
	std::string get_key();
};

template <class T>
class neuron {
	std::vector<T> signals;	// Vector to store arriving signals
	std::vector<float> weights;	// Weights of connections
	float sum;		// Weighted sum of a neuron
	float a_coeff;	// Sigmoid slope coeffitient
	float out;		// Output signal of a neuron
	float delta;	// Error of the neuron
	int ncon;		// Number of connections this neuron has

	// Summing the weighted inputs
	float calc_sum();

	// Sigmoid activation function
	// 'a' is a slope coefficient
	float activate(float a = 1.f);
public:
	neuron();
	neuron(int n);

	// Initialize connections
	void init_connections(int n);

	// Input for a new signal
	void pulse(std::vector<T>* sigs);
	void pulse(std::vector<float>* sigs);

	// Train neuron (with sigmoid 
	// slope coeffitient -> 'a')
	// 'calc_sum()' + 'activate(a)'
	void train(float a = 1.f);

	// Get the neuron output
	float get_output();
	// Get weights vector
	std::vector<float>* get_weights();

	void set_delta(float d);
	void adjust_weights(std::vector<float>* prev_out);
	void adjust_weights(std::vector<uint8_t>* prev_out);
};

template <class T>
class perceptron3 {
	// Layers of the perceptron
	std::vector<neuron<T>> l1;
	std::vector<neuron<T>> l2;
	std::vector<neuron<T>> l3;

	// Layer-output vectors
	std::vector<float> l1_out;
	std::vector<float> l2_out;
	std::vector<float> l3_out;

	// Error vector amd the
	// general error of the network
	std::vector<float> vec_err;
	float general_error;

	// Class map
	// Desired result map (key is string and value is a set of outputs)
	std::unordered_map<std::string, std::vector<float>> desresult;
public:
	perceptron3(int n1, int n2, int n3);
	perceptron3(int n1, int n2, int n3, int trainset_sz);

	// Initialize neurons by giving each one of them the
	// number of connections, which leads to generating
	// corresponding amount of wheights (ramdomized)
	void init_neurons(int sample_sz, int l1_sz, int l2_sz);
	// Initializes the class-map of desired results with
	// keys being the label for a sample and value -
	// a vector length 'n3' (number of neurons in the output layer)
	// with correponding neurons being adjusted for this class of 
	// input sample.
	void init_trainset(std::vector<sample>* trainset);

	// One epoch training
	void teach_1e(std::vector<sample>* trainset);
};
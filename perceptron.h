#pragma once
#include <vector>
#include <unordered_map>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>
#include <fstream>

class sample {
	std::vector<uint8_t> values;
	std::string key;
public:
	sample();
	sample(std::string arg_key);

	// Get a pointer to a value-vector of this sample
	std::vector<uint8_t>* const get_pvalues();
	std::vector<uint8_t>& get_rvalues();
	// Get size of a values vector (size of sample)
	int get_size();
	// Get key
	std::string get_key();
};

template <class T>
class neuron {
	std::vector<float> signals;	// Vector to store arriving signals
	std::vector<float> weights;	// Weights of connections
	float sum;		// Weighted sum of a neuron
	float a_coeff;	// Sigmoid slope coeffitient
	float out;		// Output signal of a neuron
	float error;	// Error of the neuron
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
	void init_connections(int n, float divider = 1.f);

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

	void set_err(float err);
	void adjust_weights();
	void adjust_weights_d(float delta);
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
	float cost;

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

	// Training function
	void teach(std::vector<sample>* trainset, int n_epochs);
};

// --------------------------
// Class 'sample' definitions
// --------------------------

// PUBLIC

sample::sample() {
	key = "%none%";
}

sample::sample(std::string arg_key) {
	key = arg_key;
}

// Get a pointer to a value-vector of this sample
std::vector<uint8_t>* const sample::get_pvalues() {
	return &values;
}

inline std::vector<uint8_t>& sample::get_rvalues()
{
	return values;
}

int sample::get_size() {
	return values.size();
}

std::string sample::get_key() {
	return key;
}

// --------------------------
// Class 'neuron' definitions
// --------------------------

// PRIVATE

// Summing the weighted inputs
template <class T>
float neuron<T>::calc_sum() {
	sum = 0;
	for (int i = 0; i < ncon; i++) {
		sum += float(signals[i]) * weights[i];
	}

	return sum;
}

// Sigmoid activation function
// 'a' is a slope coefficient
template <class T>
float neuron<T>::activate(float a) {
	a_coeff = a;
	out = 1 / (1 + exp(-a_coeff * sum));
	return out;
}

// PUBLIC

template <class T>
neuron<T>::neuron() {
	out = 0;
	sum = 0;
	ncon = 0;
	error = 0;
	a_coeff = 1;
}

template <class T>
neuron<T>::neuron(int n) {
	out = 0;
	sum = 0;
	ncon = n;
}

// Initialize connections
template <class T>
void neuron<T>::init_connections(int n, float divider) { // Divider parameter??
	ncon = n;
	for (int i = 0; i < ncon; i++) {
		this->weights.push_back(float(rand()) / RAND_MAX / divider /* / divider ???*/);
	}
}

// Input for a new signal
template <class T>
void neuron<T>::pulse(std::vector<T>* sigs) {
	signals.clear(); ncon = 0;
	for (T el : *sigs) {
		signals.push_back(float(el));
		ncon++;
	}
}

template <class T>
void neuron<T>::pulse(std::vector<float>* sigs) {
	signals.clear(); ncon = 0;
	for (float el : *sigs) {
		signals.push_back(el);
		ncon++;
	}
}

template <class T>
void neuron<T>::train(float a) {
	a_coeff = a;
	calc_sum();
	activate(a_coeff);
}

// Get the neuron output
template <class T>
float neuron<T>::get_output() {
	return out;
}

template<class T>
std::vector<float>* neuron<T>::get_weights()
{
	return &weights;
}

template<class T>
void neuron<T>::set_err(float err)
{
	error = err;
}

template<class T>
void neuron<T>::adjust_weights()
{
	float t_speed = 0.1;
	float derv = out * (1 - out);
	float delta = error * derv;

	int i = 0;
	for (auto& w : weights) {
		w -= t_speed * delta * float(signals.at(i));
		i++;
	}
}

template<class T>
inline void neuron<T>::adjust_weights_d(float delta)
{
	float t_speed = 0.1;
	float derv = out * (1 - out);

	int i = 0;
	for (auto& w : weights) {
		w -= t_speed * delta * derv * float(signals.at(i));
		i++;
	}
}

// ------------------------------
// Class 'perceptron' definitions
// ------------------------------

template<class T>
perceptron3<T>::perceptron3(int n1, int n2, int n3)
{
	// Initializing layer vectors
	for (int i = 0; i < n1; i++)
		l1.push_back(neuron<T>());
	for (int i = 0; i < n2; i++)
		l2.push_back(neuron<T>());
	for (int i = 0; i < n3; i++)
		l3.push_back(neuron<T>());

	cost = 0.f;
}

template<class T>
perceptron3<T>::perceptron3(int n1, int n2, int n3, int trainset_sz)
{
	// Initializing layer vectors
	for (int i = 0; i < n1; i++)
		l1.push_back(neuron<T>(trainset_sz));
	for (int i = 0; i < n2; i++)
		l2.push_back(neuron<T>(n1));
	for (int i = 0; i < n3; i++)
		l3.push_back(neuron<T>(n2));

	cost = 0.f;
}

template<class T>
void perceptron3<T>::init_neurons(int sample_sz, int l1_sz, int l2_sz)
{
	for (auto& el : l1) {
		el.init_connections(sample_sz);
	}
	for (auto& el : l2) {
		el.init_connections(l1_sz);
	}
	for (auto& el : l3) {
		el.init_connections(l2_sz);
	}
}

template<class T>
void perceptron3<T>::init_trainset(std::vector<sample>* trainset)
{
	// Mapping a class to a desired network output
	int i = 0;
	std::vector<float> outs;
	for (sample t_sample : *trainset) {
		outs.clear();
		for (int j = 0; j < l3.size(); j++) {
			if (j == i) {
				outs.push_back(0.999f);
			}
			else {
				outs.push_back(0.001f);
			}
		}
		desresult.emplace(t_sample.get_key(), outs);
		i++;
	}
}

template<class T>
void perceptron3<T>::teach(std::vector<sample>* trainset, int n_epochs) {
	float ideal = 0.f;
	std::vector<float> loss;
	float e = 0.f;
	float sum = 0.f;

	// Vector of error and deltas 
	// for each neuron in each layer
	std::vector<float> l1_err, l2_err, l3_err;
	std::vector<float> deltas3, deltas2, deltas1;

	//std::ofstream log("log.txt");

	for (int epoch = 1; epoch <= n_epochs; epoch++) {
		//log << "--- EPOCH [" << epoch << "] ---" << std::endl;
		
		cost = 0.f;
		for (sample t_sample : *trainset) {
			// Clearing layer-output vectors
			l1_out.clear(); l2_out.clear(); l3_out.clear();
			// Clearing layer-deltas vectors
			deltas3.clear(); deltas2.clear(); deltas1.clear();
			// Clearing layer-error vectors
			l1_err.clear(); l2_err.clear(); l3_err.clear();
			// Clearing vector of differences on output layer
			loss.clear();
			
			// - Forward propagation of the training sample -
			for (auto& neuron : l1) {
				neuron.pulse(t_sample.get_pvalues());
				neuron.train(1.f);

				l1_out.push_back(neuron.get_output());
			}
			for (auto& neuron : l2) {
				neuron.pulse(&l1_out);
				neuron.train(1.f);

				l2_out.push_back(neuron.get_output());
			}
			for (auto& neuron : l3) {
				neuron.pulse(&l2_out);
				neuron.train(0.3f);

				l3_out.push_back(neuron.get_output());
				std::cout << "O: " << l3_out.back() << std::endl;
				//log << l3_out.back() << std::endl;
			} //log << "---------------------" << std::endl;
			std::cout << "---------------------\n" << std::endl;
			// - Back propagation -
			// Calculating an error
			int j, i = 0;
			for (float actual_out : l3_out) {
				ideal = desresult[t_sample.get_key()].at(i);
				loss.push_back(actual_out - ideal);
				
				cost += loss.back() * loss.back();
				i++;
			}

			// Calculating errors of layer3
			i = 0;
			for (auto& neuron : l3) {
				l3_err.push_back(loss.at(i));
				neuron.set_err(loss.at(i));
				deltas3.push_back(l3_err.back() * l3_out.at(i) * (1 - l3_out.at(i)));
				i++;
			}

			// Calculating errors of layer 2
			i = 0;
			for (auto& neuron : l2) {
				sum = 0.f;
				j = 0;
				for (float delta : deltas3) {
					sum += delta * l3[j].get_weights()->at(i);
					j++;
				}
				l2_err.push_back(sum);
				neuron.set_err(sum);
				deltas2.push_back(l2_err.back() * l2_out.at(i) * (1 - l2_out.at(i)));
				i++;
			}

			// Calculating errors of layer 1
			i = 0;
			for (auto& neuron : l1) {
				sum = 0.f;
				j = 0;
				for (float delta : deltas2) {
					sum += delta * l2[j].get_weights()->at(i);
					j++;
				}
				l1_err.push_back(sum);
				neuron.set_err(sum);
				deltas1.push_back(l1_err.back() * l1_out.at(i) * (1 - l1_out.at(i)));
				i++;
			}

			// - Changing weights -
			for (auto& neuron : l3) {
				neuron.adjust_weights();
			}
			for (auto& neuron : l2) {
				neuron.adjust_weights();
			}
			for (auto& neuron : l1) {
				neuron.adjust_weights();
			}
		}

		//log << "E: " << sqrt(cost / trainset->size()) << "\n" << std::endl;
		std::cout << "E: " << sqrt(cost / trainset->size()) << "\n" << std::endl;
	}

	//log.close();
}

/*
// Calculating errors of layer3
			int j; i = 0;
			for (auto& neuron : l3) {
				l3_err.push_back(loss.at(i));
				neuron.set_err(loss.at(i));
				deltas3.push_back(l3_err.back() * l3_out.at(i) * (1 - l3_out.at(i)));
				i++;
			}

			// Calculating errors of layer 2
			i = 0;
			for (auto& neuron : l2) {
				sum = 0.f;
				j = 0;
				for (float delta : deltas3) {
					sum += delta * l3[j].get_weights()->at(i);
					j++;
				}
				l2_err.push_back(sum);
				neuron.set_err(sum);
				deltas2.push_back(l2_err.back() * l2_out.at(i) * (1 - l2_out.at(i)));
				i++;
			}

			// Calculating errors of layer 1
			i = 0;
			for (auto& neuron : l1) {
				sum = 0.f;
				j = 0;
				for (float delta : deltas2) {
					sum += delta * l2[j].get_weights()->at(i);
					j++;
				}
				l1_err.push_back(sum);
				neuron.set_err(sum);
				deltas1.push_back(l1_err.back() * l1_out.at(i) * (1 - l1_out.at(i)));
				i++;
			}

			// - Changing weights -
			for (auto& neuron : l3) {
				neuron.adjust_weights();
			}
			for (auto& neuron : l2) {
				neuron.adjust_weights();
			}
			for (auto& neuron : l1) {
				neuron.adjust_weights();
			}*/



/*

*/
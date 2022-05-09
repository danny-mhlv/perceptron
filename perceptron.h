#pragma once
#include <vector>
#include <unordered_map>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <random>

#include "matplotlibcpp.h"

namespace mpp = matplotlibcpp;

class sample {
	std::vector<std::vector<uint8_t>> values;
	std::string key;
public:
	sample();
	sample(std::string arg_key);

	// Get a pointer to a value-vector of this sample
	std::vector<std::vector<uint8_t>>* const get_pvalues();
	std::vector<std::vector<uint8_t>>& get_rvalues();
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
	void init_input(int n);

	// Input for a new signal
	void pulse(std::vector<T>* sigs);
	void pulse(std::vector<float>* sigs);

	// Train neuron (with sigmoid 
	// slope coeffitient -> 'a')
	// 'calc_sum()' + 'activate(a)'
	void train(float a = 1.f);

	// Get the neuron output
	float get_output();
	// Get number of connections
	int get_ncon();
	// Get weights vector
	const std::vector<float> const* get_weights();
	const std::vector<float> const* get_signals();

	void adjust_weights(std::vector<float>* gradients, float rate);
};

template <class T>
class perceptron3 {
	// Layers of the perceptron
	std::vector<neuron<T>> l1;
	std::vector<neuron<T>> l2;
	std::vector<neuron<T>> l3;

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
std::vector<std::vector<uint8_t>>* const sample::get_pvalues() {
	return &values;
}

inline std::vector<std::vector<uint8_t>>& sample::get_rvalues()
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
	this->sum = 0;
	for (int i = 0; i < ncon; i++) {
		this->sum += float(signals.at(i)) * weights.at(i);
	}

	return this->sum;
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
	a_coeff = 1;
}

template <class T>
neuron<T>::neuron(int n) {
	out = 0;
	sum = 0;
	ncon = n;
	a_coeff = 1;
}

// Initialize connections
template <class T>
void neuron<T>::init_connections(int n, float divider) { 
	ncon = n;
	for (int i = 0; i < ncon; i++) {
		this->weights.push_back(float(rand()) / RAND_MAX / divider);
	}
}

template <class T>
void neuron<T>::init_input(int n) {
	ncon = n;
	for (int i = 0; i < ncon; i++) {
		this->weights.push_back(1);
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
inline int neuron<T>::get_ncon()
{
	return ncon;
}

template<class T>
const std::vector<float> const* neuron<T>::get_weights()
{
	return &weights;
}

template<class T>
const std::vector<float> const* neuron<T>::get_signals()
{
	return &signals;
}

template<class T>
inline void neuron<T>::adjust_weights(std::vector<float>* gradients, float rate)
{
	int i = 0;
	for (float& w : weights) {
		w = w - rate * gradients->at(i);
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
				outs.push_back(0.9999f);
			}
			else {
				outs.push_back(0.1000f);
			}
		}
		desresult.emplace(t_sample.get_key(), outs);
		i++;
	}
}

template<class T>
void perceptron3<T>::teach(std::vector<sample>* trainset, int n_epochs) {
	std::vector<float> errs;
	
	float ideal = 0.f;
	float sum = 0.f;
	float cost_of_sample = 0.f;
	std::vector<float> costs;
	float grad_sum = 0.f;
	int neuron_cnt = 0;
	int sample_cnt = 0;

	// Output vectors
	std::vector<float> l1_out, l2_out, l3_out;
	// Delta vectors
	std::vector<float> deltas3, deltas2, deltas1;
	// Gradient matricies
	std::vector<std::vector<std::vector<float>>>
		grad3(l3.size(), std::vector<std::vector<float>>(trainset->size(), std::vector<float>(0))),
		grad2(l2.size(), std::vector<std::vector<float>>(trainset->size(), std::vector<float>(0))),
		grad1(l1.size(), std::vector<std::vector<float>>(trainset->size(), std::vector<float>(0)));
	// Averaged grads for 1 neuron
	std::vector<float> avg_gradients;

	for (int epoch = 1; epoch <= n_epochs; epoch++) {
		std::cout << "-- EPOCH " << epoch << " --" << std::endl;
		// Clearing and re-instancing gradient matricies
		grad3.clear(); grad2.clear(); grad1.clear();
		grad3.resize(l3.size());
		for (auto& el : grad3) {
			el.resize(trainset->size());
		}
		grad2.resize(l2.size());
		for (auto& el : grad2) {
			el.resize(trainset->size());
		}
		grad1.resize(l1.size());
		for (auto& el : grad1) {
			el.resize(trainset->size());
		}

		// Shuffles the trainset each epoch
		std::shuffle(trainset->begin(), trainset->end(), std::random_device());
		
		sample_cnt = 0;
		for (sample t_sample : *trainset) {
			deltas3.clear(); deltas2.clear(); deltas1.clear();	// Clearing layer-deltas vectors
			l1_out.clear(); l2_out.clear(); l3_out.clear();		// Clearing layer-output vectors
			
			// - Forward propagation of the training sample -
			std::vector<std::vector<uint8_t>> image;
			int slice = 0;
			for (neuron<T>& n : l1) {
				image.push_back(t_sample.get_pvalues()->at(slice));
				n.pulse(&(t_sample.get_pvalues()->at(slice)));
				n.train(1.f);

				l1_out.push_back(n.get_output());
				slice++;
			} //display_cimg(&image);
			for (neuron<T>& n : l2) {
				n.pulse(&l1_out);
				n.train(1.f);

				l2_out.push_back(n.get_output());
			} std::cout << t_sample.get_key() << std::endl;
			for (neuron<T>& n : l3) {
				n.pulse(&l2_out);
				n.train(0.6f);
				
				l3_out.push_back(n.get_output());
				std::cout << l3_out.back() << std::endl;
			} std::cout << "---------------------------------\n" << std::endl;

			// - Back propagation -
			
			int i = 0, j = 0;
			cost_of_sample = 0.f;
			for (float actual_out : l3_out) {
				ideal = desresult[t_sample.get_key()].at(i);
				cost_of_sample += (actual_out - ideal) * (actual_out - ideal);
				i++;
			}

			i = 0; j = 0;
			for (float actual_out : l3_out) {	// Calculating deltas of l3
				ideal = desresult[t_sample.get_key()].at(i);

				deltas3.push_back((actual_out - ideal));
				i++;
			}
			
			i = 0, j = 0;
			for (float output : l2_out) {		// Calculating deltas of l2
				sum = 0.f; i = 0;
				for (neuron<T>& n : l3) {
					sum += deltas3.at(i) * n.get_weights()->at(j);
					i++;
				}
				deltas2.push_back(sum * output * (1 - output));
				j++;
			}
			
			i = 0, j = 0;
			for (float output : l1_out) {		// Calculating deltas of l1
				sum = 0.f; i = 0;
				for (neuron<T>& n : l2) {
					sum += deltas2.at(i) * n.get_weights()->at(j);
					i++;
				}
				deltas1.push_back(sum * output * (1 - output));
				j++;
			}

			// Calculating gradients for l3
			neuron_cnt = 0;
			for (neuron<T>& n : l3) {
				for (int gradient_cnt = 0; gradient_cnt < n.get_ncon(); gradient_cnt++) {
					grad3.at(neuron_cnt).at(sample_cnt)
						.push_back(deltas3.at(neuron_cnt) * n.get_signals()->at(gradient_cnt));
				}
				neuron_cnt++;
			}
			// Calculating gradients for l2
			neuron_cnt = 0;
			for (neuron<T>& n : l2) {
				for (int gradient_cnt = 0; gradient_cnt < n.get_ncon(); gradient_cnt++) {
					grad2.at(neuron_cnt).at(sample_cnt)
						.push_back(deltas2.at(neuron_cnt) * n.get_signals()->at(gradient_cnt));
				}
				neuron_cnt++;
			}
			// Calculating gradients for l1
			neuron_cnt = 0;
			for (neuron<T>& n : l1) {
				for (int gradient_cnt = 0; gradient_cnt < n.get_ncon(); gradient_cnt++) {
					grad1.at(neuron_cnt).at(sample_cnt)
						.push_back(deltas1.at(neuron_cnt) * n.get_signals()->at(gradient_cnt));
				}
				neuron_cnt++;
			}
			sample_cnt++;
		}
		
		neuron_cnt = 0;
		for (neuron<T>& n : l3) {	// Averaging gradients of layer 3
			avg_gradients.clear();

			for (int gradient_cnt = 0; gradient_cnt < n.get_ncon(); gradient_cnt++) {
				grad_sum = 0.f;
				for (int sample_cnt = 0; sample_cnt < grad3.at(neuron_cnt).size(); sample_cnt++) {
					grad_sum += grad3.at(neuron_cnt).at(sample_cnt).at(gradient_cnt);
				}
				avg_gradients.push_back(grad_sum / trainset->size());
			}

			n.adjust_weights(&avg_gradients, 0.1);
			neuron_cnt++;
		}

		neuron_cnt = 0;
		for (neuron<T>& n : l2) {	// Averaging gradients of layer 3
			avg_gradients.clear();

			for (int gradient_cnt = 0; gradient_cnt < n.get_ncon(); gradient_cnt++) {
				grad_sum = 0.f;
				for (int sample_cnt = 0; sample_cnt < grad2.at(neuron_cnt).size(); sample_cnt++) {
					grad_sum += grad2.at(neuron_cnt).at(sample_cnt).at(gradient_cnt);
				}
				avg_gradients.push_back(grad_sum / trainset->size());
			}

			n.adjust_weights(&avg_gradients, 0.1);
			neuron_cnt++;
		}

		neuron_cnt = 0;
		for (neuron<T>& n : l1) {	// Averaging gradients of layer 3
			avg_gradients.clear();

			for (int gradient_cnt = 0; gradient_cnt < n.get_ncon(); gradient_cnt++) {
				grad_sum = 0.f;
				for (int sample_cnt = 0; sample_cnt < grad1.at(neuron_cnt).size(); sample_cnt++) {
					grad_sum += grad1.at(neuron_cnt).at(sample_cnt).at(gradient_cnt);
				}
				avg_gradients.push_back(grad_sum / trainset->size());
			}

			n.adjust_weights(&avg_gradients, 0.1);
			neuron_cnt++;
		}

		errs.push_back(cost_of_sample / trainset->size());
	}

	mpp::figure_size(800, 600);
	mpp::xlim(0, n_epochs);
	mpp::title("Error");
	mpp::plot(errs);
	mpp::show();
}
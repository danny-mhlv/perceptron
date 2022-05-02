#include "perceptron.h"

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
template <class T >
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
}

template <class T>
neuron<T>::neuron(int n) {
	out = 0;
	sum = 0;
	ncon = n;
}

// Initialize connections
template <class T>
void neuron<T>::init_connections(int n) {
	ncon = n;
	for (int i = 0; i < ncon; i++) {
		this->weights.push_back(float(rand()) / RAND_MAX);
	}
}

// Input for a new signal
template <class T>
void neuron<T>::pulse(std::vector<T>* sigs) {
	signals.clear(); ncon = 0;
	for (T el : *sigs) {
		signals.push_back(el);
		ncon++;
	}
}

template <class T>
void neuron<T>::pulse(std::vector<float>* sigs) {
	signals.clear(); ncon = 0;
	for (float el : *sigs) {
		signals.push_back((T)el);
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
void neuron<T>::set_delta(float d)
{
	delta = d;
}

template<class T>
void neuron<T>::adjust_weights(std::vector<float>* prev_out)
{
	float t_speed = 0.1;
	float der = 1 / (1 + exp(-a_coeff * sum)) *
		(1 - (1 / (1 + exp(-a_coeff * sum))));
	int i = 0;
	for (auto& w : weights) {
		w -= -0.1 * delta * der * prev_out[i];
		i++;
	}
}

template<class T>
void neuron<T>::adjust_weights(std::vector<uint8_t>* prev_out)
{
	float t_speed = 0.1;
	float der = 1 / (1 + exp(-a_coeff * sum)) *
		(1 - (1 / (1 + exp(-a_coeff * sum))));
	int i = 0;
	for (auto& w : weights) {
		w -= -0.1 * delta * der * float(prev_out[i]);
		i++;
	}
}

// ------------------------------
// Class 'perceptron' definitions
// ------------------------------

template <class T>
perceptron3<T>::perceptron3(int n1, int n2, int n3) {
	// Initializing layer vectors
	for (int i = 0; i < n1; i++)
		l1.push_back(neuron<T>());
	for (int i = 0; i < n2; i++)
		l2.push_back(neuron<T>());
	for (int i = 0; i < n3; i++)
		l3.push_back(neuron<T>());

	general_error = 0.f;
}

template<class T>
perceptron3<T>::perceptron3(int n1, int n2, int n3, int trainset_sz) {
	// Initializing layer vectors
	for (int i = 0; i < n1; i++)
		l1.push_back(neuron<T>(trainset_sz));
	for (int i = 0; i < n2; i++)
		l2.push_back(neuron<T>(n1));
	for (int i = 0; i < n3; i++)
		l3.push_back(neuron<T>(n2));

	general_error = 0f;
}

template <class T>
void perceptron3<T>::init_neurons(int sample_sz, int l1_sz, int l2_sz) {
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

template <class T>
void perceptron3<T>::init_trainset(std::vector<sample>* trainset) {
	// Mapping a class to a desired network output
	int i = 0;
	std::vector<float> outs;
	for (sample t_sample : *trainset) {
		outs.clear();
		for (int j = 0; j < l3.size(); j++) {
			if (j == i) {
				outs.push_back(0.9f);
			}
			else {
				outs.push_back(0.1f);
			}
		}
		desresult.emplace(t_sample.get_key(), outs);
		i++;
	}
}

// One epoch training
template <class T>
void perceptron3<T>::teach_1e(std::vector<sample>* trainset) {
	float actual = 0.f;
	float ideal = 0.f;
	std::vector<float> diff;
	float e = 0.f;
	float sum = 0.f;

	// Vector of deltas for each neuron
	// in each layer
	std::vector<float> d1;
	std::vector<float> d2;
	std::vector<float> d3;

	for (sample t_sample : *trainset) {
		// Clearing output vectors on a new sample
		// introduction to the network
		l1_out.clear();	d1.clear();
		l2_out.clear(); d2.clear();
		l3_out.clear(); d3.clear();
		diff.clear();
		vec_err.clear();

		// Forward propagation
		for (neuron<T> neuron : l1) {
			neuron.pulse(t_sample.get_pvalues());
			neuron.train();
			l1_out.push_back(neuron.get_output());
		}
		for (neuron<T> neuron : l2) {
			std::vector<T> tmp;
			neuron.pulse(&l1_out);
			neuron.train();
			l2_out.push_back(neuron.get_output());
		}
		for (neuron<T> neuron : l3) {
			neuron.pulse(&l2_out);
			neuron.train(3.f);
			l3_out.push_back(neuron.get_output());
		}

		// Calculating an error
		for (int i = 0; i < l3_out.size(); i++) {
			actual = l3_out[i];
			ideal = desresult[t_sample.get_key()][i];
			diff.push_back(actual - ideal); // Difference between ideal and actual output
			e = (diff.back() * diff.back()) / 2;
			vec_err.push_back(e);
			general_error += e;
		}

		// Back propagation
		int i = 0;
		for (auto& neuron : l3) {
			neuron.set_delta(diff[i] * l3_out[i] * (1 - l3_out[i]));
			neuron.adjust_weights(&l2_out);
			i++;
		}

		// Caculating deltas for the second layer
		for (int i = 0; i < l2.size(); i++) {
			sum = 0.f;
			for (int j = 0; j < d3.size(); j++) {
				sum += d3[j] * l3[j].weights[i];
			}
			d2.push_back(sum);
		}
		i = 0;
		for (auto& neuron : l2) {
			neuron.set_delta(d2[i]);
			neuron.adjust_weights(&l1_out);
			i++;
		}

		// Caculating deltas for the first layer
		for (int i = 0; i < l1.size(); i++) {
			sum = 0.f;
			for (int j = 0; j < d2.size(); j++) {
				sum += d2[j] * l2[j].weights[i];
			}
			d1.push_back(sum);
		}
		i = 0;
		for (auto& neuron : l1) {
			neuron.set_delta(d1[i]);
			neuron.adjust_weights(t_sample.get_pvalues());
			i++;
		}
	}
}
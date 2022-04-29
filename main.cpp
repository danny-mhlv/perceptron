#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <ctime>
#include <string>

/*template <class T>
class sample {
	std::vector<T> 
public:

};*/

template <class T>
class neuron {
	std::vector<T> signals;	// Vector to store arriving signals
	std::vector<float> weights;	// Weights of connections
	float sum;	// Weighted sum of a neuron
	float out;	// Output signal of a neuron
	int ncon;	// Number of connections this neuron has

	// Summing the weighted inputs
	float calc_sum() {
		sum = 0;
		for (int i; i < ncon; i++) {
			sum += float(signals[i]) * weights[i];
		}

		return sum;
	}

	// Sigmoid activation function
	// 'a' is a slope coefficient
	float activate(float a) {
		out = 1 / (1 + exp(-a * sum));
		return out;
	}
public:
	neuron() {
		out = 0;
		sum = 0;
		ncon = 0;
	}

	// Initialize connections
	void init_connections(int n) {
		ncon = n;
	}

	// Input for a new signal
	void pulse(std::vector<T>* sigs) {
		signals.clear(); ncon = 0;
		for (T el : *sigs) {
			signals.push_back(el);
			ncon++;
		}
	}

	// Get the neuron output
	float output() {
		return out;
	}
};

template <class T>
class perceptron3 {
	// Layers of the perceptron
	std::vector<neuron<T>> l1;
	std::vector<neuron<T>> l2;
	std::vector<neuron<T>> l3;

	// Class map
	std::unordered_map<char, std::vector<float>> desresult; // Desired result map (key is char and value is a set of outputs)

	// Error vector
	std::vector<float> vec_err;
public:
	perceptron3(int n1, int n2, int n3) {
		// Initializing layer vectors
		for (int i = 0; i < n1; i++)
			l1.push_back(neuron<T>());
		for (int i = 0; i < n2; i++)
			l2.push_back(neuron<T>());
		for (int i = 0; i < n3; i++)
			l3.push_back(neuron<T>());
	}

	void teach(std::vector<float/*sample*/>* trainset) {

	}

	void calc_error(/*vector of */) {
		for (neuron<T> el : l3) {
			vec_err.push_back(el.output());
		}
	}
};

uint8_t grayscale(uint8_t r, uint8_t g, uint8_t b, uint8_t maxval) {
	float fr = r / maxval, fg = g / maxval, fb = b / maxval;
	uint8_t _r, _g, _b, _y, y;

	_r = (fr <= 0.04045) ? (fr / 12.92) : pow((fr / 0.055) / 1.055, 2.4);
	_g = (fg <= 0.04045) ? (fg / 12.92) : pow((fg / 0.055) / 1.055, 2.4);
	_b = (fb <= 0.04045) ? (fb / 12.92) : pow((fb / 0.055) / 1.055, 2.4);

	_y = 0.2126 * _r + 0.7152 * _g + 0.0722 *_b;
	if (_y <= 0.0031308) {
		y = 12.92 * _y;
		return uint8_t(round(y * 255));
	}
	else {
		y = 1.055 * pow(_y, 1/2.4) - 0.055;
		return uint8_t(round(y * 255));
	}
}

// Reads Netpbm file of any type and packs into
// 'vec' vector in 1/0 format [bw = Black/White]
void read_netpbm_bw(const char* filename, std::vector<uint8_t>* img) {
	char btype[3] = { 0 };					// File type

	char bw[5] = { 0 }; char bh[5] = { 0 }; // Width and height of image
	int w, h;

	char maxval[4] = { 0 };					// Max color value

	char sym;								// For ASCII-based files
	uint8_t val;							// For binary files
	
	std::ifstream file(filename, std::ios::binary);
	// Reading header
	file.getline(btype, 3);
	file.getline(bw, 5, ' ');
	file.getline(bh, 5);
	std::string type(btype);
	w = atoi(bw);
	h = atoi(bh);

	// P1-6 is the magic number and file type indicator:
	// ------------------------------- | 1 and 4 -> .pbm (b/w)
	// 1-3 | is plain text (ASCII)	   | 2 and 5 -> .pgm (0-255 gray scale)
	// 4-6 | is raw (binary)		   | 3 and 6 -> .ppm (3 channels 0-255 each)
	
	if (type == "P1") {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				file.read(&sym, 1);
				img->push_back(atoi(&sym));
			}
		}
	}
	else if (type == "P2") {
		file.getline(maxval, 4);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				file.read(&sym, 1);
			}
		}
	}
	else if (type == "P3") {
		file.getline(maxval, 4);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

			}
		}
		file.read(&sym, 1);
	}
	else if (type == "P4") {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

			}
		}
		file.read(reinterpret_cast<char*>(&val), 1);
	}
	else if (type == "P5") {
		file.getline(maxval, 4);

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {

			}
		}
		file.read(reinterpret_cast<char*>(&val), 1);
	}
	else if (type == "P6") {
		file.getline(maxval, 4);

		uint8_t r, g, b;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				file.read(reinterpret_cast<char*>(&r), 1);
				file.read(reinterpret_cast<char*>(&g), 1);
				file.read(reinterpret_cast<char*>(&b), 1);

				val = grayscale(r, g, b, 255);

				(val < 233) ? val = 0 : val = 1;

				img->push_back(val);
			}
		}
	}
	else {
		std::cout << "Not a Netpbm file or header corrupted!" << std::endl;
		return;
	}

	// Display
	int n = 0;
	for (uint8_t el : *img) {
		if (n % w == 0) {
			std::cout << "\n";
		}
		if (el == 1) {
			std::cout << "@";
		}
		else {
			std::cout << ".";
		}
		n++;
	}

	file.close();
}

int main(int argc, char* argv[]) {
	//perceptron3<float> net(6, 8, 4);
	std::vector<uint8_t> vec;
	read_netpbm_bw("./imgsrc/s1.ppm", &vec);


	return 0;
}
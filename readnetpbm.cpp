#include "readnetpbm.h"

uint8_t grayscale(uint8_t r, uint8_t g, uint8_t b, uint8_t maxval) {
	float fr = (float)r / maxval, fg = (float)g / maxval, fb = (float)b / maxval;
	float _r, _g, _b, _y, y;

	_r = (fr <= 0.04045f) ? (fr / 12.92f) : (float)pow((fr + 0.055f) / 1.055f, 2.4);
	_g = (fg <= 0.04045f) ? (fg / 12.92f) : (float)pow((fg + 0.055f) / 1.055f, 2.4);
	_b = (fb <= 0.04045f) ? (fb / 12.92f) : (float)pow((fb + 0.055f) / 1.055f, 2.4);

	_y = 0.2126f * _r + 0.7152f * _g + 0.0722f * _b;
	if (_y <= 0.00313) {
		y = 12.92f * _y;
		return uint8_t(round(y * maxval));
	}
	else {
		y = 1.055f * (float)pow(_y, 1 / 2.4) - 0.055f;
		return uint8_t(round(y * maxval));
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

				val = grayscale(r, g, b, 255);	//Grayscaling image
				(val < 128) ? val = 0 : val = 1; // Turning it black/white

				img->push_back(val);
			}
		}
	}
	else {
		std::cout << "Not a Netpbm file or header corrupted!" << std::endl;
		return;
	}

	file.close();
}

/*// Display
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
	}*/
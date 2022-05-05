#include <vector>
#include <discpp.h>
#include "CImg.h"
#include "readnetpbm.h"
#include "perceptron.h"

void display_cimg(std::vector<uint8_t>* img, int w, int h, int pixelsz);

int main(int argc, char* argv[]) {
	// Init random seed
	srand((unsigned)time(0));

	// Init trainset
	std::vector<sample> trainset;
	trainset.push_back(sample("Bobsled"));
	read_netpbm_bw("./imgsrc/s1.ppm", trainset.back().get_pvalues());
	//display_cimg(trainset.back().get_pvalues(), 54, 54, 1);
	trainset.push_back(sample("Slide"));
	read_netpbm_bw("./imgsrc/s2.ppm", trainset.back().get_pvalues());
	//display_cimg(trainset.back().get_pvalues(), 54, 54, 1);
	trainset.push_back(sample("Skates"));
	read_netpbm_bw("./imgsrc/s3.ppm", trainset.back().get_pvalues());
	//display_cimg(trainset.back().get_pvalues(), 54, 54, 1);
	trainset.push_back(sample("Fight"));
	read_netpbm_bw("./imgsrc/s4.ppm", trainset.back().get_pvalues());
	//display_cimg(trainset.back().get_pvalues(), 54, 54, 1);

	// Init network
	const int layer1 = 6, layer2 = 4, layer3 = 4;
	perceptron3<uint8_t> net(layer1, layer2, layer3);
	net.init_neurons(trainset.back().get_size(), layer1, layer2);
	net.init_trainset(&trainset);

	// Generate noisy images
	/*for (int i = 0; i < 2; i++) {
		trainset.push_back(sample("Bobsled"));
		read_netpbm_bw("./imgsrc/s1.ppm", trainset.back().get_pvalues());
		apply_noise(trainset.back().get_pvalues(), 20);
	}

	for (int i = 0; i < 2; i++) {
		trainset.push_back(sample("Slide"));
		read_netpbm_bw("./imgsrc/s2.ppm", trainset.back().get_pvalues());
		apply_noise(trainset.back().get_pvalues(), 20);
	}

	for (int i = 0; i < 2; i++) {
		trainset.push_back(sample("Skates"));
		read_netpbm_bw("./imgsrc/s3.ppm", trainset.back().get_pvalues());
		apply_noise(trainset.back().get_pvalues(), 20);
	}

	for (int i = 0; i < 2; i++) {
		trainset.push_back(sample("Fight"));
		read_netpbm_bw("./imgsrc/s4.ppm", trainset.back().get_pvalues());
		apply_noise(trainset.back().get_pvalues(), 20);
		
	}*/

	net.teach(&trainset, 500);

	return 0;
}

void display_cimg(std::vector<uint8_t>* img, int w, int h, int pixelsz) {
	using namespace cimg_library;
	const uint8_t* const data = &(img->at(0));
	CImg<uint8_t> cimg(data, w, h);
	cimg.display();
}

/*
void display_cimg(std::vector<uint8_t>* img, int w, int h, int pixelsz) {
	using namespace cimg_library;
	CImg<uint8_t> cimg(w, h, 1, 1, 0);

	int posx, posy;
	uint8_t color;

	int i = 0, j = 0;
	for (uint8_t pixel : *img) {
		if (i != 0 && i % w == 0) {
			j++;
		}

		posx = i * pixelsz;
		posy = j * pixelsz;
		color = pixel * 255;
		//cimg.draw_rectangle(posx, posy, posx + pixelsz, posy + pixelsz, color);

		i++;
	}

	cimg.display();
}*/
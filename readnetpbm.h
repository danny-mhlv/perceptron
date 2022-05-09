#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

// Grayscales the 24-bit pixel
uint8_t grayscale(uint8_t r, uint8_t g, uint8_t b, uint8_t maxval);

// Reads Netpbm file of any type and packs into
// 'vec' vector in 1/0 format [bw = Black/White]
void read_netpbm_bw(const char* filename, 
	std::vector<uint8_t>* img);
void read_netpbm_bw(const char* filename, 
	std::vector<std::vector<uint8_t>>* img);

// Applies noise by inverting b/w pixels at a certain rate
void apply_noise(std::vector<uint8_t>* img, float percent);

// Displays a b/w image in ASCII-format
void display_ascii(std::vector<uint8_t>* img, int w);
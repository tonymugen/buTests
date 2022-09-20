/*
 * Copyright (c) 2022 Anthony J. Greenberg
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <array>
#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <random>

#include "../externals/bayesicUtilities/random.hpp"

uint64_t ranIntLooped() noexcept {
	unsigned long long rInt;
	// this can be in infinite loop if the random number pool is depleted
	// but then we have more serious system-wide issues
	while (_rdrand64_step(&rInt) == 0){
	}

	return static_cast<uint64_t> (rInt);
}

int32_t ranIntUnchk(unsigned long long &rInt) noexcept {
	return _rdrand64_step(&rInt);
}

int main(int argc, char *argv[]){
	std::chrono::duration<float, std::milli> loopTime;
	std::chrono::duration<float, std::milli> unchkTime;
	std::chrono::duration<float, std::milli> baseTime;
	std::chrono::duration<float, std::milli> classTime;
	std::chrono::duration<float, std::milli> stdTime;
	uint64_t tstVal;
	unsigned long long tstULL;
	auto time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 9999; ++i){
		tstVal = ranIntLooped();
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	loopTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 9999; ++i){
		const int32_t trash = ranIntUnchk(tstULL);
		tstVal = static_cast<uint64_t> (tstULL);
	}
	time2 = std::chrono::high_resolution_clock::now();
	unchkTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 9999; ++i){
		const int32_t trash = _rdrand64_step(&tstULL);
		tstVal = static_cast<uint64_t> (tstULL);
	}
	time2 = std::chrono::high_resolution_clock::now();
	baseTime = time2 - time1;
	BayesicSpace::RanDraw tstRan;
	std::array<uint64_t, 10000> res;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 9999; ++i){
		res[i] = tstRan.ranInt();
	}
	time2 = std::chrono::high_resolution_clock::now();
	classTime = time2 - time1;
	std::mt19937_64 stdGen;
	stdGen.seed( __rdtsc() );
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 9999; ++i){
		res[i] = stdGen();
	}
	time2 = std::chrono::high_resolution_clock::now();
	stdTime = time2 - time1;
	
	std::cout << loopTime.count() << " " << unchkTime.count() << " " << baseTime.count() << " " << classTime.count() << " | " << stdTime.count() << "\n";
}

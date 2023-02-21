/*
 * Copyright (c) 2023 Anthony J. Greenberg
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

/** \file Tests of PRNG correctness
 *
 * Testing development versions of the PRNG library against main.
 *
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "random.hpp"

int main(int argc, char *argv[]) {
	uint64_t seed{0};
	if (argc > 1){
		seed = static_cast<uint64_t>( std::stoi( std::string(argv[1]) ) );
	} else {
		BayesicSpace::RanDraw loc;
		seed = loc.ranInt();
	}
	BayesicSpace::RanDraw testObj(seed);
	constexpr size_t cycleCount{100};
	constexpr uint64_t max{10000};
	constexpr uint64_t min{100};
	constexpr double sigma{2.0};
	constexpr double mean{3.0};
	constexpr double alpha{5.0};
	constexpr double beta{4.0};
	constexpr double dFrdm{10.0};
	constexpr double nVit{15.0};
	constexpr double totNvit{10000.0};
	for (size_t iCycle = 0; iCycle < cycleCount; ++iCycle){
		// single-sample functions
		std::cout << "RanInt\t"          << testObj.ranInt()               << "\n";
		std::cout << "sampleIntMax\t"    << testObj.sampleInt(max)         << "\n";
		std::cout << "sampleIntMinMax\t" << testObj.sampleInt(min, max)    << "\n";
		std::cout << "runif\t"           << testObj.runif()                << "\n";
		std::cout << "runifnz\t"         << testObj.runifnz()              << "\n";
		std::cout << "runifno\t"         << testObj.runifno()              << "\n";
		std::cout << "runifop\t"         << testObj.runifop()              << "\n";
		std::cout << "rnorm\t"           << testObj.rnorm()                << "\n";
		std::cout << "rnormSig\t"        << testObj.rnorm(sigma)           << "\n";
		std::cout << "rnormMnSig\t"      << testObj.rnorm(mean, sigma)     << "\n";
		std::cout << "rgamma\t"          << testObj.rgamma(alpha)          << "\n";
		std::cout << "rgammaScaled\t"    << testObj.rgamma(alpha, beta)    << "\n";
		std::cout << "rchisq\t"          << testObj.rchisq(dFrdm)          << "\n";
		std::cout << "vitterA\t"         << testObj.vitterA(nVit, totNvit) << "\n";
		std::cout << "vitter\t"          << testObj.vitter(nVit, totNvit)  << "\n";
		// Dirichlet with vectors
		std::vector<double> aVec{1.5, 2.0, 2.5, 3.0, 3.5};
		std::vector<double> pVec(5, 0.0);
		testObj.rdirichlet(aVec, pVec);
		for (const auto &pEl : pVec){
			std::cout << "rdirichlet\t" << pEl << "\n";
		}
	}
}


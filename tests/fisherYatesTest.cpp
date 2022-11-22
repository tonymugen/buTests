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
#include <vector>
#include <chrono>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>

#include "random.hpp"

uint64_t moduloInt(BayesicSpace::RanDraw &r, const uint64_t &max) { return r.ranInt() % max; }
uint64_t canonInt(BayesicSpace::RanDraw &r, const uint64_t &max) {
	const __uint128_t max128 = static_cast<__uint128_t>(max);
	const __uint128_t r0     = static_cast<__uint128_t>( r.ranInt() ) * max128;
	const uint64_t r1Hi      = static_cast<uint64_t>( (static_cast<__uint128_t>( r.ranInt() ) * max128) >> 64 );
	const uint64_t r0Lo      = static_cast<uint64_t>(r0);
	const uint64_t r0Hi      = static_cast<uint64_t>(r0 >> 64);
	const uint64_t sumHi     = static_cast<uint64_t>( ( static_cast<__uint128_t>(r0Lo) + static_cast<__uint128_t>(r1Hi) ) >> 64 );
	//sumHi                    = static_cast<uint64_t>( static_cast<__uint128_t>(r0Hi) + static_cast<__uint128_t>(sumHi) );
	return static_cast<uint64_t>( static_cast<__uint128_t>(r0Hi) + static_cast<__uint128_t>(sumHi) );
}

uint64_t canonInt(std::mt19937_64 &r, const uint64_t &max) {
	const __uint128_t max128 = static_cast<__uint128_t>(max);
	const uint64_t v1        = r();
	const __uint128_t r0     = static_cast<__uint128_t>(v1) * max128;
	const uint64_t v2        = r();
	const uint64_t r1Hi      = static_cast<uint64_t>( (static_cast<__uint128_t>(v2) * max128) >> 64 );
	const uint64_t r0Lo      = static_cast<uint64_t>(r0);
	const uint64_t r0Hi      = static_cast<uint64_t>(r0 >> 64);
	const uint64_t sumHi     = static_cast<uint64_t>( ( static_cast<__uint128_t>(r0Lo) + static_cast<__uint128_t>(r1Hi) ) >> 64 );
	//sumHi                    = static_cast<uint64_t>( static_cast<__uint128_t>(r0Hi) + static_cast<__uint128_t>(sumHi) );
	return static_cast<uint64_t>( static_cast<__uint128_t>(r0Hi) + static_cast<__uint128_t>(sumHi) );
}

uint64_t lemireInt(BayesicSpace::RanDraw &r, const uint64_t &max){
	uint64_t x    = r.ranInt();
	__uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(max);
	uint64_t l    = static_cast<uint64_t>(m);
	if (l < max){
		const uint64_t t = static_cast<uint64_t>(-max) % max;
		while (l < t){
			x = r.ranInt();
			m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(max);
			l = static_cast<uint64_t>(m);
		}
	}

	return static_cast<uint64_t>(m >> 64);
}

uint64_t lemireInt(std::mt19937_64 &r, const uint64_t &max){
	uint64_t x    = r();
	__uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(max);
	uint64_t l    = static_cast<uint64_t>(m);
	if (l < max){
		const uint64_t t = static_cast<uint64_t>(-max) % max;
		while (l < t){
			x = r();
			m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(max);
			l = static_cast<uint64_t>(m);
		}
	}

	return static_cast<uint64_t>(m >> 64);
}


int main(){
	BayesicSpace::RanDraw tstRun;
	std::mt19937_64 stdGen;
	stdGen.seed( tstRun.ranInt() );
	//std::vector<uint64_t> res{tstRun.shuffleUintDown(1000)};
	//std::vector<uint64_t> res{tstRun.shuffleUintUp(1000)};
	/*
	std::array<uint64_t, 2001> res{0};
	for (size_t i = 0; i < 200100000000; ++i){
		++res[tstRun.sampleInt(2001)];
	}
	for (const auto ir : res){
		std::cout << ir << " ";
	}
	*/
	//std::cout << canonInt(stdGen, 20) << "\n";
	std::cout << canonInt(tstRun, 20) << "\n";
	std::cout << lemireInt(tstRun, 20) << "\n";
	//std::cout << lemireInt(stdGen, 20) << "\n";
	/*
	std::chrono::duration<float, std::milli> upTime;
	std::chrono::duration<float, std::milli> downTime;
	auto time1 = std::chrono::high_resolution_clock::now();
	for (size_t l = 0; l < 50000; ++l){
		std::vector<uint64_t> loc{tstRun.shuffleUintUp(1000)};
		res.emplace_back(loc[0]);
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	upTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t l = 0; l < 50000; ++l){
		std::vector<uint64_t> loc{tstRun.shuffleUintDown(1000)};
		res.emplace_back(loc[0]);
	}
	time2 = std::chrono::high_resolution_clock::now();
	downTime = time2 - time1;
	std::cout << upTime.count() << "\tUp1000\tyes\n" << downTime.count() << "\tDown1000\tyes\n";
	*/
	return 0;
}

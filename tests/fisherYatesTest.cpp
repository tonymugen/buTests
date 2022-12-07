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
#include <numeric>
#include <vector>
#include <chrono>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
#include <iterator>

#include "random.hpp"

uint64_t moduloInt(BayesicSpace::RanDraw &r, const uint64_t &max) { return r.ranInt() % max; }
uint64_t canonInt(BayesicSpace::RanDraw &r, const uint64_t &max) {
	// This method is described in
	// https://github.com/apple/swift/pull/39143
	// https://jacquesheunis.com/post/bounded-random/
	// A Swift implementation is in
	// https://github.com/stephentyrone/swift/blob/played-for-absolute-fools/stdlib/public/core/Random.swift
	const __uint128_t max128 = static_cast<__uint128_t>(max);
	const __uint128_t r0     = static_cast<__uint128_t>( r.ranInt() ) * max128;
	const __uint128_t r1Hi   = (static_cast<__uint128_t>( r.ranInt() ) * max128) >> 64;
	const uint64_t r0Lo      = static_cast<uint64_t>(r0);
	const __uint128_t r0Hi   = r0 >> 64;
	const __uint128_t sumHi  = (static_cast<__uint128_t>(r0Lo) + r1Hi) >> 64;
	return static_cast<uint64_t>(r0Hi + sumHi);
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

int main(){
	BayesicSpace::RanDraw tstRun;
	//std::vector<uint64_t> res{tstRun.shuffleUintDown(100)};
	//std::vector<uint64_t> res{tstRun.shuffleUintUp(1000)};
	constexpr size_t N = 23;
	for (size_t k = 0; k < 10000000; ++k){
		std::vector<size_t> perInd{tstRun.fyIndexesUp(N)};
		std::vector<uint64_t> res(N);
		std::iota(res.begin(), res.end(), 1);
		for (size_t i = 0; i < N - 1; ++i){
			std::swap(res[i], res[ perInd[i] ]);
		}
		for (const auto ir : res){
			std::cout << ir << " ";
		}
		std::cout << "\n";
	}
	return 0;
}

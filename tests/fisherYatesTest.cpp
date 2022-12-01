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
	//std::vector<uint64_t> res{tstRun.shuffleUintDown(1000)};
	//std::vector<uint64_t> res{tstRun.shuffleUintUp(1000)};
	std::chrono::duration<float, std::micro> moduloTime;
	std::chrono::duration<float, std::micro> lemireTime;
	std::chrono::duration<float, std::micro> canonTime;
	/*
	uint64_t lo = 0;
	uint64_t hi = 0;
	auto time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 100000; ++i){
		const uint64_t x = moduloInt(tstRun, 4294867296);
		if (x <= 1410265407){
			++lo;
		} else {
			++hi;
		}
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	moduloTime = time2 - time1;
	lo = 0;
	hi = 0;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 100000; ++i){
		const uint64_t x = lemireInt(tstRun, 4294867296);
		if (x <= 1410265407){
			++lo;
		} else {
			++hi;
		}
	}
	time2 = std::chrono::high_resolution_clock::now();
	lemireTime = time2 - time1;
	lo = 0;
	hi = 0;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 100000; ++i){
		const uint64_t x = canonInt(tstRun, 4294867296);
		if (x <= 1410265407){
			++lo;
		} else {
			++hi;
		}
	}
	time2 = std::chrono::high_resolution_clock::now();
	canonTime = time2 - time1;
	std::cout << moduloTime.count() * 1e-5 << "\tmodulo\n" << lemireTime.count() * 1e-5 << "\tlemire\n" << canonTime.count() * 1e-5 << "\tcanon\n";
	*/
	std::cout << lemireInt(tstRun, 0) << "\n";
	std::vector<uint64_t> out(100000);
	std::iota(out.begin(), out.end(), 0);
	auto time1 = std::chrono::high_resolution_clock::now();
	for (uint64_t i = 99999; i > 0; --i){
		uint64_t j = moduloInt(tstRun, i + 1);
		out[i] ^= out[j];
		out[j] ^= out[i];
		out[i] ^= out[j];
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	moduloTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (uint64_t i = 99999; i > 0; --i){
		uint64_t j = lemireInt(tstRun, i + 1);
		out[i] ^= out[j];
		out[j] ^= out[i];
		out[i] ^= out[j];
	}
	time2 = std::chrono::high_resolution_clock::now();
	lemireTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (uint64_t i = 99999; i > 0; --i){
		uint64_t j = canonInt(tstRun, i + 1);
		out[i] ^= out[j];
		out[j] ^= out[i];
		out[i] ^= out[j];
	}
	time2 = std::chrono::high_resolution_clock::now();
	canonTime = time2 - time1;
	std::cout << moduloTime.count() << "\tmodulo\n" << lemireTime.count() << "\tlemire\n" << canonTime.count() << "\tcanon\n";
	BayesicSpace::RanDraw one(11);
	BayesicSpace::RanDraw two(11);
	for (size_t i = 0; i < 25; ++i){
		std::cout << one.sampleInt(2001) << " " << lemireInt(two, 2001) << "\n";
	}
	return 0;
}

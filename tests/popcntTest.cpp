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

/** \file Checks the popcnt instruction
 *
 * This file runs checks the popcnt instruction results against a processor agnostic method.
 *
 */

#include <cstdint>
#include <vector>
#include <limits>
#include <iostream>
#include <cstring>
#include <chrono>
#include <immintrin.h>

#include "random.hpp"

constexpr size_t wordSize    = 8;
constexpr uint64_t roundMask = 0xfffffffffffffff8;

uint64_t countSetBits(const std::vector<uint8_t> &inVec) {
	uint64_t totSet{0};
	for (auto inv : inVec){
		for (; static_cast<bool>(inv); ++totSet) {
			inv &= inv - 1;
		}
	}
	return totSet;
}

uint64_t countSetBitsPOP(const std::vector<uint8_t> &inVec){
	uint64_t totSet{0};
	const size_t nWholeWords = inVec.size() & roundMask;
	size_t iByte{0};
	while (iByte < nWholeWords){
		uint64_t chunk{0};
		memcpy(&chunk, inVec.data() + iByte, wordSize);
		totSet += static_cast<uint64_t>( _mm_popcnt_u64(chunk) );
		iByte += wordSize;
	}
	if ( nWholeWords < inVec.size() ){
		uint64_t chunk{0};
		memcpy(&chunk, inVec.data() + iByte, inVec.size() - nWholeWords);
		totSet += static_cast<uint64_t>( _mm_popcnt_u64(chunk) );
	}
	return totSet;
}

int main(){
	std::chrono::duration<float, std::milli> karnTime{0};
	std::chrono::duration<float, std::milli> popcntTime{0};
	constexpr size_t nBytes{100003};
	constexpr size_t nWords = nBytes / wordSize + static_cast<size_t>(nBytes % wordSize > 0);
	std::cout << "nBytes = " << nBytes << "; nWords = " << nWords << "\n";
	BayesicSpace::RanDraw seedPRNG;
	std::vector<uint64_t> ranBits;
	for (size_t i = 0; i < nWords; ++i){
		ranBits.push_back( seedPRNG.ranInt() );
	}
	std::vector<uint8_t> binBytes( nBytes, std::numeric_limits<uint8_t>::max() );
	size_t iByte{0};
	for (const auto eachWord : ranBits){
		for (size_t byteInWord = 0; (byteInWord < wordSize) && (iByte < nBytes); ++byteInWord){
			binBytes[iByte] = static_cast<uint8_t>( eachWord >> (byteInWord * wordSize) ) & (static_cast<uint8_t>(1) << byteInWord);
			++iByte;
		}
	}
	auto time1               = std::chrono::high_resolution_clock::now();
	const uint64_t karnRes   = countSetBits(binBytes);
	auto time2               = std::chrono::high_resolution_clock::now();
	karnTime                 = time2 - time1;
	time1                    = std::chrono::high_resolution_clock::now();
	const uint64_t popcntRes = countSetBitsPOP(binBytes);
	time2                    = std::chrono::high_resolution_clock::now();
	popcntTime               = time2 - time1;
	std::cout << karnRes << " " << popcntRes << "\n";
	std::cout << karnTime.count() << " " << popcntTime.count() << "\n";
}


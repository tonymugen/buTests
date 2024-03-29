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
#include <numeric>
#include <vector>
#include <limits>
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <functional>
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

uint64_t countEqual(const std::vector<uint16_t> &vec1, const std::vector<uint16_t> &vec2) {
	uint64_t sum{0};
	for (size_t iVec = 0; iVec < vec1.size(); ++iVec){
		sum += static_cast<uint64_t>(vec1[iVec] == vec2[iVec]);
	}
	return sum;
}

uint64_t countEqualDP(const std::vector<uint16_t> &vec1, const std::vector<uint16_t> &vec2) {
	const uint64_t sum = static_cast<uint64_t>( std::inner_product( vec1.begin(), vec1.end(), vec2.begin(), 0, std::plus<>(), std::equal_to<>() ) );
	return sum;
}

uint64_t countEqualAVX(const std::vector<uint16_t> &vec1, const std::vector<uint16_t> &vec2) {
	constexpr size_t bytesInVec = 32;
	constexpr size_t elsInVec   = 16;
	constexpr auto roundMask16  = static_cast<uint64_t>(-16);
	uint64_t sum{0};
	size_t iChunk{0};
	const size_t nWholeVec{vec1.size() & roundMask16};
	while (iChunk < nWholeVec){
		__m256i vec1p{0};
		__m256i vec2p{0};
		memcpy(&vec1p, vec1.data() + iChunk, bytesInVec);
		memcpy(&vec2p, vec2.data() + iChunk, bytesInVec);
		const __m256i resVec{_mm256_cmpeq_epi16(vec1p, vec2p)};
		auto res64 = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 0) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		res64      = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 1) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		res64      = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 2) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		res64      = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 3) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		iChunk    += elsInVec;
	}
	if (vec1.size() > nWholeVec){
		__m256i vec1p{0};
		__m256i vec2p{_mm256_set1_epi32(-1)};  // set all bits so that the tails are unequal
		const size_t remainder = vec1.size() - nWholeVec;
		memcpy(&vec1p, vec1.data() + iChunk, remainder);
		memcpy(&vec2p, vec2.data() + iChunk, remainder);
		const __m256i resVec{_mm256_cmpeq_epi16(vec1p, vec2p)};
		auto res64 = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 0) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		res64      = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 1) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		res64      = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 2) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
		res64      = static_cast<uint64_t>( _mm256_extract_epi64(resVec, 3) );
		sum       += static_cast<uint64_t>( _mm_popcnt_u64(res64) );
	}
	sum = sum / elsInVec;
	return sum;
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
	std::cout << "=====================\nCMPEQ test\n";
	std::chrono::duration<float, std::milli> sumTime{0};
	std::chrono::duration<float, std::milli> dpTime{0};
	std::chrono::duration<float, std::milli> avxTime{0};
	constexpr size_t tstVecSize = 1600003;
	std::vector<uint16_t> tstVec1(tstVecSize, 1);
	std::vector<uint16_t> tstVec2(tstVecSize, 3);
	tstVec1[100] = tstVec2[100];
	tstVec1[303] = tstVec2[303];
	time1                  = std::chrono::high_resolution_clock::now();
	const uint64_t tstRes1 = countEqual(tstVec1, tstVec2);
	time2                  = std::chrono::high_resolution_clock::now();
	sumTime                = time2 - time1;
	time1                  = std::chrono::high_resolution_clock::now();
	const uint64_t tstRes2 = countEqualDP(tstVec1, tstVec2);
	time2                  = std::chrono::high_resolution_clock::now();
	dpTime                 = time2 - time1;
	time1                  = std::chrono::high_resolution_clock::now();
	const uint64_t tstRes3 = countEqualAVX(tstVec1, tstVec2);
	time2                  = std::chrono::high_resolution_clock::now();
	avxTime                = time2 - time1;
	std::cout << tstRes1 << " " << tstRes2 << " " << tstRes3 << "\n";
	std::cout << sumTime.count() << " " << dpTime.count() << " " << avxTime.count() << "\n";
}


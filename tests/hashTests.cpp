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

#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <limits>
#include <array>
#include <bitset>

#include "random.hpp"

uint32_t murMurHash(const size_t &key, const uint32_t &seed) {
	uint32_t hash = seed;

	// body
	auto blocks = reinterpret_cast<const uint32_t *>(&key);

	for (size_t i = 0; i < sizeof(size_t) / sizeof(uint32_t); ++i){
		uint32_t k1 = blocks[i];

		k1 *= 0xcc9e2d51;
		k1  = (k1 << 15) | (k1 >> 17);
		k1 *= 0x1b873593;

		hash ^= k1;
		hash  = (hash << 13) | (hash >> 19);
		hash  = hash * 5 + 0xe6546b64;
	}

	// there is no tail since the input is fixed (at eight bytes typically)
	// finalize
	hash ^= sizeof(size_t);
	hash ^= hash >> 16;
	hash *= 0x85ebca6b;
	hash ^= hash >> 13;
	hash *= 0xc2b2ae35;
	hash ^= hash >> 16;

	return hash;
}

std::array<float, 2> locusOPH(const size_t &locusInd, const size_t &nIndividuals, const size_t &locusSize, const size_t &kSketches, const size_t &sketchSize,
	const std::vector<size_t> &permutation, std::vector<uint32_t> &seeds, BayesicSpace::RanDraw &rng, std::vector<uint8_t> &binLocus, std::vector<uint16_t> &sketches) {
	std::chrono::duration<float, std::milli> permTime;
	std::chrono::duration<float, std::milli> sketchTime;
	// Start with a permutation to make OPH
	auto time1 = std::chrono::high_resolution_clock::now();
	const uint8_t byteSize       = 8;
	//const uint8_t oneBit         = 0b00000001;
	//const uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	size_t iIndiv                = nIndividuals - 1UL;
	for (const auto &ri : permutation){
		const uint16_t firstIdx = iIndiv % byteSize;
		const size_t firstByte  = iIndiv / byteSize;
		uint16_t secondIdx      = ri % byteSize;
		const size_t secondByte = ri / byteSize;
		const uint16_t diff     = byteSize * (firstByte != secondByte); // will be 0 if the same byte is being accessed; then need to swap bits within byte

		// swapping bits within a two-byte variable
		// using the method in https://graphics.stanford.edu/~seander/bithacks.html#SwappingBitsXOR
		// if the same byte is being accessed, secondIdx is not shifted to the second byte
		// This may be affected by endianness (did not test)
		uint16_t twoBytes  = (static_cast<uint16_t>(binLocus[secondByte]) << 8) | ( static_cast<uint16_t>(binLocus[firstByte]) );
		secondIdx         += diff;
		uint16_t x         = ( (twoBytes >> firstIdx) ^ (twoBytes >> secondIdx) ) & 1;
		twoBytes          ^= ( (x << firstIdx) | (x << secondIdx) );

		memcpy( binLocus.data() + firstByte, &twoBytes, sizeof(uint8_t) );
		twoBytes = twoBytes >> diff;
		memcpy( binLocus.data() + secondByte, &twoBytes, sizeof(uint8_t) );
		--iIndiv;
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	permTime = time2 - time1;
	/*
	// Now make the sketches
	time1 = std::chrono::high_resolution_clock::now();
	std::vector<size_t> filledIndexes;                // indexes of the non-empty sketches
	size_t iByte     = 0;
	size_t colEnd    = iByte + locusSize;
	size_t sketchBeg = locusInd * kSketches;
	size_t iSeed     = 0;                             // index into the seed vector
	uint8_t iInByte  = 0;
	// A possible optimization is to test a whole byte for 0
	// Will test later
	for (size_t iSketch = 0; iSketch < kSketches; ++iSketch){
		uint16_t firstSetBitPos = 0;
		while ( (iByte != colEnd) && ( ( (oneBit << iInByte) & binLocus[iByte] ) == 0 ) &&
				(firstSetBitPos < sketchSize) ){
			++iInByte;
			// these are instead of an if statement
			iByte  += iInByte == byteSize;
			iInByte = iInByte % byteSize;
			++firstSetBitPos;
		}
		if ( (iByte < colEnd) && (firstSetBitPos < sketchSize) ){
			filledIndexes.push_back(iSketch);
			{
				// should be safe: each thread accesses different vector elements
				sketches[sketchBeg + iSketch] = firstSetBitPos;
			}

			uint16_t remainder = sketchSize - firstSetBitPos;
			uint16_t inByteMod = remainder % byteSize;
			uint16_t inByteSum = iInByte + inByteMod;

			iByte  += remainder / byteSize + inByteSum / byteSize;
			iInByte = inByteSum % byteSize;
		}
	}
	if (filledIndexes.size() == 1){
		for (size_t iSk = 0; iSk < kSketches; ++iSk){ // this will overwrite the one assigned sketch, but the wasted operation should be swamped by the rest
			// should be safe: each thread accesses different vector elements
			sketches[sketchBeg + iSk] = sketches[filledIndexes[0] + sketchBeg];
		}
	} else if (filledIndexes.size() != kSketches){
		if ( filledIndexes.empty() ){ // in the case where the whole locus is monomorphic, pick a random index as filled
			filledIndexes.push_back( rng.sampleInt(kSketches) );
		}
		size_t emptyCount = kSketches - filledIndexes.size();
		while (emptyCount){
			for (const auto &f : filledIndexes){
				uint32_t newIdx = murMurHash(f, seeds[iSeed]) % kSketches + sketchBeg;
				// should be safe: each thread accesses different vector elements
				if (sketches[newIdx] == emptyBinToken){
					sketches[newIdx] = sketches[f + sketchBeg];
					--emptyCount;
					break;
				}
			}
			++iSeed;
			if ( iSeed == seeds.size() ){
				seeds.push_back( static_cast<uint32_t>( rng.ranInt() ) );
			}
		}
	}
	time2      = std::chrono::high_resolution_clock::now();
	*/
	sketchTime = time2 - time1;
	return std::array<float, 2>{permTime.count(), sketchTime.count()};
}

std::array<float, 2> locusOPHnew(const size_t &locusInd, const size_t &nIndividuals, const size_t &locusSize, const size_t &kSketches, const size_t &sketchSize,
	const std::vector<size_t> &permutation, std::vector<uint32_t> &seeds, BayesicSpace::RanDraw &rng, std::vector<uint8_t> &binLocus, std::vector<uint16_t> &sketches) {
	std::chrono::duration<float, std::milli> permTime;
	std::chrono::duration<float, std::milli> sketchTime;
	// Start with a permutation to make OPH
	auto time1 = std::chrono::high_resolution_clock::now();
	const uint8_t byteSize       = 8;
	//const uint8_t oneBit         = 0b00000001;
	//const uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	for (size_t iIndiv = 0; iIndiv < nIndividuals - 1; ++iIndiv){
		const uint16_t firstIdx = iIndiv % byteSize;
		const size_t firstByte  = iIndiv / byteSize;
		uint16_t secondIdx      = permutation[iIndiv] % byteSize;
		const size_t secondByte = permutation[iIndiv] / byteSize;
		const uint16_t diff     = byteSize * (firstByte != secondByte); // will be 0 if the same byte is being accessed; then need to swap bits within byte

		// swapping bits within a two-byte variable
		// using the method in https://graphics.stanford.edu/~seander/bithacks.html#SwappingBitsXOR
		// if the same byte is being accessed, secondIdx is not shifted to the second byte
		// This may be affected by endianness (did not test)
		uint16_t twoBytes  = (static_cast<uint16_t>(binLocus[secondByte]) << 8) | ( static_cast<uint16_t>(binLocus[firstByte]) );
		secondIdx         += diff;
		uint16_t x         = ( (twoBytes >> firstIdx) ^ (twoBytes >> secondIdx) ) & 1;
		twoBytes          ^= ( (x << firstIdx) | (x << secondIdx) );

		memcpy( binLocus.data() + firstByte, &twoBytes, sizeof(uint8_t) );
		twoBytes = twoBytes >> diff;
		memcpy( binLocus.data() + secondByte, &twoBytes, sizeof(uint8_t) );
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	permTime = time2 - time1;
	/*
	// Now make the sketches
	time1 = std::chrono::high_resolution_clock::now();
	std::vector<size_t> filledIndexes;                // indexes of the non-empty sketches
	size_t iByte     = 0;
	size_t colEnd    = iByte + locusSize;
	size_t sketchBeg = locusInd * kSketches;
	size_t iSeed     = 0;                             // index into the seed vector
	uint8_t iInByte  = 0;
	// A possible optimization is to test a whole byte for 0
	// Will test later
	for (size_t iSketch = 0; iSketch < kSketches; ++iSketch){
		uint16_t firstSetBitPos = 0;
		while ( (iByte != colEnd) && ( ( (oneBit << iInByte) & binLocus[iByte] ) == 0 ) &&
				(firstSetBitPos < sketchSize) ){
			++iInByte;
			// these are instead of an if statement
			iByte  += iInByte == byteSize;
			iInByte = iInByte % byteSize;
			++firstSetBitPos;
		}
		if ( (iByte < colEnd) && (firstSetBitPos < sketchSize) ){
			filledIndexes.push_back(iSketch);
			{
				// should be safe: each thread accesses different vector elements
				sketches[sketchBeg + iSketch] = firstSetBitPos;
			}

			uint16_t remainder = sketchSize - firstSetBitPos;
			uint16_t inByteMod = remainder % byteSize;
			uint16_t inByteSum = iInByte + inByteMod;

			iByte  += remainder / byteSize + inByteSum / byteSize;
			iInByte = inByteSum % byteSize;
		}
	}
	if (filledIndexes.size() == 1){
		for (size_t iSk = 0; iSk < kSketches; ++iSk){ // this will overwrite the one assigned sketch, but the wasted operation should be swamped by the rest
			// should be safe: each thread accesses different vector elements
			sketches[sketchBeg + iSk] = sketches[filledIndexes[0] + sketchBeg];
		}
	} else if (filledIndexes.size() != kSketches){
		if ( filledIndexes.empty() ){ // in the case where the whole locus is monomorphic, pick a random index as filled
			filledIndexes.push_back( rng.sampleInt(kSketches) );
		}
		size_t emptyCount = kSketches - filledIndexes.size();
		while (emptyCount){
			for (const auto &f : filledIndexes){
				uint32_t newIdx = murMurHash(f, seeds[iSeed]) % kSketches + sketchBeg;
				// should be safe: each thread accesses different vector elements
				if (sketches[newIdx] == emptyBinToken){
					sketches[newIdx] = sketches[f + sketchBeg];
					--emptyCount;
					break;
				}
			}
			++iSeed;
			if ( iSeed == seeds.size() ){
				seeds.push_back( static_cast<uint32_t>( rng.ranInt() ) );
			}
		}
	}
	time2      = std::chrono::high_resolution_clock::now();
	*/
	sketchTime = time2 - time1;
	return std::array<float, 2>{permTime.count(), sketchTime.count()};
}

int main() {
	BayesicSpace::RanDraw prng;
	//const size_t nIndividuals    = 1200;
	const size_t nIndividuals    = 125;
	//const size_t kSketches       = 100;
	const size_t kSketches       = 20;
	//const size_t locusSize       = nIndividuals / 8 + static_cast<bool>(nIndividuals % 8);
	//const size_t sketchSize      = nIndividuals / kSketches + static_cast<bool>(nIndividuals % kSketches);
	const uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	const size_t ranBitVecSize   = nIndividuals / (sizeof(uint64_t) * 8) + static_cast<bool> ( nIndividuals % (sizeof(uint64_t) * 8) );
	std::vector<uint32_t> seeds{static_cast<uint32_t>( prng.ranInt() )};
	std::vector<size_t> ranInts{prng.shuffleUintUp(nIndividuals)};
	//std::vector<size_t> ranInts{prng.shuffleUintDown(nIndividuals)};
	std::vector<uint16_t> sketches1(kSketches, emptyBinToken);
	std::vector<uint16_t> sketches2(kSketches, emptyBinToken);
	std::vector<uint8_t> binLocus1;
	std::vector<uint8_t> binLocus2;
	std::vector<uint64_t> ranBits;
	for (size_t i = 0; i < ranBitVecSize; ++i){
		ranBits.push_back( prng.ranInt() );
	}
	for (const auto rb : ranBits){
		const auto bits = reinterpret_cast<const uint8_t *>(&rb);
		for (size_t ii = 0; ii < sizeof(uint64_t); ++ii){
			binLocus2.push_back(bits[ii]);
		}
	}
	binLocus2.back() = binLocus2.back() >> 3;
	binLocus1        = binLocus2;
	std::cout << "x64:   " << std::bitset<8>(ranBits[0]) << " " << std::bitset<8>(ranBits[0] >> 8) << " " << std::bitset<8>(ranBits[0] >> 16) << " " << std::bitset<8>(ranBits[0] >> 24) << " " << std::bitset<8>(ranBits[0] >> 32) << 
		" " << std::bitset<8>(ranBits[0] >> 40) << " " << std::bitset<8>(ranBits[0] >> 48) << " " << std::bitset<8>(ranBits[0] >> 56) << "\n";
	uint32_t x = ranBits[0] >> 32;
	std::cout << "x:     " << std::bitset<8>(x) << " " << std::bitset<8>(x >> 8) << " " << std::bitset<8>(x >> 16) << " " << std::bitset<8>(x >> 24) << "\n";
	uint64_t m{0};
	auto m32 = reinterpret_cast<uint32_t *>(&m);
	m32[0]   = 0b00010000'00100000'01000000'10000000;
	m32[1]   = 0b00000001'00000010'00000100'00001000;
	std::cout << "m:     " << std::bitset<8>(m32[1]) << " " << std::bitset<8>(m32[1] >> 8) << " " << std::bitset<8>(m32[1] >> 16) << " " << std::bitset<8>(m32[1] >> 24) << "\n";
	x &= m32[1];
	std::cout << "x & m: " << std::bitset<8>(x) << " " << std::bitset<8>(x >> 8) << " " << std::bitset<8>(x >> 16) << " " << std::bitset<8>(x >> 24) << "\n";
	uint32_t mk32 = ~m32[1] << 1;
	uint32_t mp32{0};
	uint32_t mv32{0};
	for (uint32_t i = 0; i < 5; ++i){
		mp32   = mk32 ^ (mk32 << 1);
		mp32   = mp32 ^ (mp32 << 2);
		mp32   = mp32 ^ (mp32 << 4);
		mp32   = mp32 ^ (mp32 << 8);
		mp32   = mp32 ^ (mp32 << 16);
		mv32   = mp32 & m32[1];
		m32[1] = (m32[1] ^ mv32) | ( mv32 >> (1 << i) );
		const uint32_t t32 = x & mv32;
		x    = (x ^ t32) | ( t32 >> (1 << i) );
		mk32 = mk32 & ~mp32;
	}
	//std::cout << "m:     " << std::bitset<8>(m32[1]) << " " << std::bitset<8>(m32[1] >> 8) << " " << std::bitset<8>(m32[1] >> 16) << " " << std::bitset<8>(m32[1] >> 24) << "\n";
	//std::cout << "mk:    " << std::bitset<8>(mk32) << " " << std::bitset<8>(mk32 >> 8) << " " << std::bitset<8>(mk32 >> 16) << " " << std::bitset<8>(mk32 >> 24) << "\n";
	//std::cout << "mp:    " << std::bitset<8>(mp32) << " " << std::bitset<8>(mp32 >> 8) << " " << std::bitset<8>(mp32 >> 16) << " " << std::bitset<8>(mp32 >> 24) << "\n";
	std::cout << "\n";
	std::cout << "m:     " << std::bitset<8>(m32[1]) << " " << std::bitset<8>(m32[1] >> 8) << " " << std::bitset<8>(m32[1] >> 16) << " " << std::bitset<8>(m32[1] >> 24) << "\n";
	std::cout << "x:     " << std::bitset<8>(x) << " " << std::bitset<8>(x >> 8) << " " << std::bitset<8>(x >> 16) << " " << std::bitset<8>(x >> 24) << "\n";
	/*
	//std::cout << "m: " << std::bitset<8>(m) << " " << std::bitset<8>(m >> 8) << " " << std::bitset<8>(m >> 16) << " " << std::bitset<8>(m >> 24) << " " << std::bitset<8>(m >> 32) << 
	//	" " << std::bitset<8>(m >> 40) << " " << std::bitset<8>(m >> 48) << " " << std::bitset<8>(m >> 56) << "\n";
	ranBits[0] &= m;
	//std::cout << std::bitset<8>(ranBits[0]) << " " << std::bitset<8>(ranBits[0] >> 8) << " " << std::bitset<8>(ranBits[0] >> 16) << " " << std::bitset<8>(ranBits[0] >> 24) << " " << std::bitset<8>(ranBits[0] >> 32) << 
	//	" " << std::bitset<8>(ranBits[0] >> 40) << " " << std::bitset<8>(ranBits[0] >> 48) << " " << std::bitset<8>(ranBits[0] >> 56) << "\n";
	uint64_t mk = ~m << 1;
	//std::cout << std::bitset<8>(mk) << " " << std::bitset<8>(mk >> 8) << " " << std::bitset<8>(mk >> 16) << " " << std::bitset<8>(mk >> 24) << " " << std::bitset<8>(mk >> 32) << 
	//	" " << std::bitset<8>(mk >> 40) << " " << std::bitset<8>(mk >> 48) << " " << std::bitset<8>(mk >> 56) << "\n";
	uint64_t mp = mk ^ (mk << 1);
	//std::cout << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
	//	" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	mp = mp ^ (mp << 2);
	//std::cout << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
	//	" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	mp = mp ^ (mp << 4);
	//std::cout << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
	//	" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	mp = mp ^ (mp << 8);
	//std::cout << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
	//	" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	mp = mp ^ (mp << 16);
	//std::cout << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
	//	" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	mp = mp ^ (mp << 32);
	uint64_t mv = mp & m;
	//std::cout << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
	//	" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	//std::cout << "\n";
	std::cout << "x & m: " << std::bitset<8>(ranBits[0]) << " " << std::bitset<8>(ranBits[0] >> 8) << " " << std::bitset<8>(ranBits[0] >> 16) << " " << std::bitset<8>(ranBits[0] >> 24) << " " << std::bitset<8>(ranBits[0] >> 32) << 
		" " << std::bitset<8>(ranBits[0] >> 40) << " " << std::bitset<8>(ranBits[0] >> 48) << " " << std::bitset<8>(ranBits[0] >> 56) << "\n";
	std::cout << "m:     " << std::bitset<8>(m) << " " << std::bitset<8>(m >> 8) << " " << std::bitset<8>(m >> 16) << " " << std::bitset<8>(m >> 24) << " " << std::bitset<8>(m >> 32) << 
		" " << std::bitset<8>(m >> 40) << " " << std::bitset<8>(m >> 48) << " " << std::bitset<8>(m >> 56) << "\n";
	std::cout << "mk:    " << std::bitset<8>(mk) << " " << std::bitset<8>(mk >> 8) << " " << std::bitset<8>(mk >> 16) << " " << std::bitset<8>(mk >> 24) << " " << std::bitset<8>(mk >> 32) << 
		" " << std::bitset<8>(mk >> 40) << " " << std::bitset<8>(mk >> 48) << " " << std::bitset<8>(mk >> 56) << "\n";
	std::cout << "mp:    " << std::bitset<8>(mp) << " " << std::bitset<8>(mp >> 8) << " " << std::bitset<8>(mp >> 16) << " " << std::bitset<8>(mp >> 24) << " " << std::bitset<8>(mp >> 32) << 
		" " << std::bitset<8>(mp >> 40) << " " << std::bitset<8>(mp >> 48) << " " << std::bitset<8>(mp >> 56) << "\n";
	std::cout << "mv:    " << std::bitset<8>(mv) << " " << std::bitset<8>(mv >> 8) << " " << std::bitset<8>(mv >> 16) << " " << std::bitset<8>(mv >> 24) << " " << std::bitset<8>(mv >> 32) << 
		" " << std::bitset<8>(mv >> 40) << " " << std::bitset<8>(mv >> 48) << " " << std::bitset<8>(mv >> 56) << "\n";
	m = (m ^ mv) | (mv >> 1);
	uint64_t t = ranBits[0] & mv;
	ranBits[0] = (ranBits[0] ^ t) | (t >> 1);
	std::cout << "\n";
	std::cout << "m:     " << std::bitset<8>(m) << " " << std::bitset<8>(m >> 8) << " " << std::bitset<8>(m >> 16) << " " << std::bitset<8>(m >> 24) << " " << std::bitset<8>(m >> 32) << 
		" " << std::bitset<8>(m >> 40) << " " << std::bitset<8>(m >> 48) << " " << std::bitset<8>(m >> 56) << "\n";
	std::cout << "x:     " << std::bitset<8>(ranBits[0]) << " " << std::bitset<8>(ranBits[0] >> 8) << " " << std::bitset<8>(ranBits[0] >> 16) << " " << std::bitset<8>(ranBits[0] >> 24) << " " << std::bitset<8>(ranBits[0] >> 32) << 
		" " << std::bitset<8>(ranBits[0] >> 40) << " " << std::bitset<8>(ranBits[0] >> 48) << " " << std::bitset<8>(ranBits[0] >> 56) << "\n";
	mk = mk & ~mp;
	std::cout << "mk:    " << std::bitset<8>(mk) << " " << std::bitset<8>(mk >> 8) << " " << std::bitset<8>(mk >> 16) << " " << std::bitset<8>(mk >> 24) << " " << std::bitset<8>(mk >> 32) << 
		" " << std::bitset<8>(mk >> 40) << " " << std::bitset<8>(mk >> 48) << " " << std::bitset<8>(mk >> 56) << "\n";
	*/
	//const std::array<float, 2> res1 = locusOPH(0, nIndividuals, locusSize, kSketches, sketchSize, ranInts, seeds, prng, binLocus1, sketches1);
	//const std::array<float, 2> res2 = locusOPHnew(0, nIndividuals, locusSize, kSketches, sketchSize, ranInts, seeds, prng, binLocus2, sketches2);
	//const std::array<float, 2> res2 = locusOPH(0, nIndividuals, locusSize, kSketches, sketchSize, ranInts, seeds, prng, binLocus2, sketches2);
	/*
	for (size_t i = 0; i < binLocus1.size(); ++i){
		std::cout << std::bitset<8>(binLocus1[i] ^ binLocus2[i]) << " ";
	}
	std::cout << "\n";
	*/
	//std::cout << res1[0] << "\t" << res1[1] << "\t" << res2[0] << "\t" << res2[1] << "\n";
}
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
#include <immintrin.h>

#include "random.hpp"

uint32_t murMurHash(const size_t &key, const uint32_t &seed) {
	uint32_t hash = seed;

	// body
	// TODO: change to memcpy; possible strict aliasing violation
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
	auto time1                   = std::chrono::high_resolution_clock::now();
	constexpr uint8_t byteSize   = 8;
	constexpr uint8_t oneBit     = 0b00000001;
	//const uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	// Round down to multiple of 8; nIndividuals - 1 because Fisher-Yates goes up to N - 2 inclusive
	const size_t fullByteNind = (nIndividuals - 1) & 0xfffffffffffffff8;
	size_t iIndiv             = 0;
	size_t iByte              = 0;
	std::array<size_t, 8> permByteInd;
	std::array<size_t, 8> permInByteInd;
	uint64_t bytesToSwap;
	uint64_t swapBitMask;
	std::array<uint8_t, byteSize> bytesToSwapArr;
	std::array<uint8_t, byteSize> swapBitMaskArr;
	while(iIndiv < fullByteNind){
		// Aggregate bytes that contain bits that need to swapped
		// with the bits in current byte into one 64-bit word.
		// Add a mask that marks bits to be swapped with each byte
		for (size_t iInByte = 0; iInByte < byteSize; ++iInByte){
			const size_t perIndiv   = permutation[iIndiv++];                     // post-increment to use current value for index first
			permByteInd[iInByte]    = perIndiv / byteSize;
			permInByteInd[iInByte]  = perIndiv - (perIndiv & 0xfffffffffffffff8);
			bytesToSwapArr[iInByte] = binLocus[ permByteInd[iInByte] ];
			swapBitMaskArr[iInByte] = static_cast<uint8_t>(oneBit << permInByteInd[iInByte]);
		}
		// Compress the bits corresponding to the set bits in the mask using PS-XOR (Chapter 7-4 of Hacker's Delight)
		memcpy(&bytesToSwap, bytesToSwapArr.data(), byteSize);
		memcpy(&swapBitMask, swapBitMaskArr.data(), byteSize);
		//const uint64_t xi = _pext_u64(bytesToSwap, swapBitMask); // this is the intrinsic function that does the same thing
		bytesToSwap &= swapBitMask;
		uint64_t m   = swapBitMask;
		uint64_t mk  = ~swapBitMask << 1;
		uint64_t mp;
		uint64_t mv;
		uint64_t tmp;
		for (uint32_t i = 0; i < 6; ++i){
			mp          = mk ^ (mk << 1);
			mp          = mp ^ (mp << 2);
			mp          = mp ^ (mp << 4);
			mp          = mp ^ (mp << 8);
			mp          = mp ^ (mp << 16);
			mp          = mp ^ (mp << 32);
			mv          = mp & m;
			m           = (m ^ mv) | ( mv >> (1 << i) );
			tmp         = bytesToSwap & mv;
			bytesToSwap = (bytesToSwap ^ tmp) | ( tmp >> (1 << i) );
			mk          = mk & ~mp;
		}
		memcpy(bytesToSwapArr.data(), &bytesToSwap, 1);
		// Swap (using the three XOR method) the current binLocus byte with the byte of the aggregate where the masked bits are now stored
		// TODO: check if using a temp is faster
		binLocus[iByte]   ^= bytesToSwapArr[0]; 
		bytesToSwapArr[0] ^= binLocus[iByte];
		binLocus[iByte]   ^= bytesToSwapArr[0]; 
		memcpy(&bytesToSwap, bytesToSwapArr.data(), 1);
		// Now expand the bits swapped from the current binLocus byte to the mask positions in the aggregate word (Chapter 7-5 of Hacker's Delight)
		std::array<uint64_t, 6> a;
		m  = swapBitMask;
		mk = ~m << 1;
		for (uint32_t i = 0; i < 6; ++i){
			mp   = mk ^ (mk << 1);
			mp   = mp ^ (mp << 2);
			mp   = mp ^ (mp << 4);
			mp   = mp ^ (mp << 8);
			mp   = mp ^ (mp << 16);
			mp   = mp ^ (mp << 32);
			mv   = mp & m;
			a[i] = mv;
			m    = (m ^ mv) | ( mv >> (1 << i) );
			mk   = mk & ~mp;
		}
		bytesToSwap  = (bytesToSwap & ~a[5]) | ( (bytesToSwap << 32) & a[5] );
		bytesToSwap  = (bytesToSwap & ~a[4]) | ( (bytesToSwap << 16) & a[4] );
		bytesToSwap  = (bytesToSwap & ~a[3]) | ( (bytesToSwap << 8) & a[3] );
		bytesToSwap  = (bytesToSwap & ~a[2]) | ( (bytesToSwap << 4) & a[2] );
		bytesToSwap  = (bytesToSwap & ~a[1]) | ( (bytesToSwap << 2) & a[1] );
		bytesToSwap  = (bytesToSwap & ~a[0]) | ( (bytesToSwap << 1) & a[0] );
		bytesToSwap &= swapBitMask;
		memcpy(bytesToSwapArr.data(), &bytesToSwap, byteSize);
		// Finally, replace the relevant bits in the binLocus bytes indexed by the permutation
		// Order of operations is important!
		// Chapter 2-20 of Hacker's Delight, but we do not need the full swap
		for (size_t inByteInd = 0; inByteInd < byteSize; ++inByteInd){
			binLocus[ permByteInd[inByteInd] ] ^= (binLocus[ permByteInd[inByteInd] ] ^ bytesToSwapArr[inByteInd]) & swapBitMaskArr[inByteInd];
		}
		++iByte;
	}
	if (iIndiv < nIndividuals - 1){
		std::cout << "tail\n";
		swapBitMaskArr.fill(0);
		const size_t remainIndiv = nIndividuals - 1 - iIndiv;
		for (size_t iRem = 0; iRem < remainIndiv; ++iRem){
			const size_t perIndiv = permutation[iIndiv++];                     // post-increment to use current value for index first
			permByteInd[iRem]     = perIndiv / byteSize;
			permInByteInd[iRem]   = perIndiv - (perIndiv & 0xfffffffffffffff8);
			bytesToSwapArr[iRem]  = binLocus[ permByteInd[iRem] ];
			swapBitMaskArr[iRem]  = static_cast<uint8_t>(oneBit << permInByteInd[iRem]);
		}
		// Compress the bits corresponding to the set bits in the mask using PS-XOR (Chapter 7-4 of Hacker's Delight)
		memcpy(&bytesToSwap, bytesToSwapArr.data(), byteSize);
		memcpy(&swapBitMask, swapBitMaskArr.data(), byteSize);
		//const uint64_t xi = _pext_u64(bytesToSwap, swapBitMask); // this is the intrinsic function that does the same thing
		bytesToSwap &= swapBitMask;
		uint64_t m   = swapBitMask;
		uint64_t mk  = ~swapBitMask << 1;
		uint64_t mp;
		uint64_t mv;
		uint64_t tmp;
		for (uint32_t i = 0; i < 6; ++i){
			mp          = mk ^ (mk << 1);
			mp          = mp ^ (mp << 2);
			mp          = mp ^ (mp << 4);
			mp          = mp ^ (mp << 8);
			mp          = mp ^ (mp << 16);
			mp          = mp ^ (mp << 32);
			mv          = mp & m;
			m           = (m ^ mv) | ( mv >> (1 << i) );
			tmp         = bytesToSwap & mv;
			bytesToSwap = (bytesToSwap ^ tmp) | ( tmp >> (1 << i) );
			mk          = mk & ~mp;
		}
		memcpy(bytesToSwapArr.data(), &bytesToSwap, 1);
		// Swap (using the three XOR method) the current binLocus byte with the byte of the aggregate where the masked bits are now stored
		// TODO: check if using a temp is faster
		binLocus[iByte]   ^= bytesToSwapArr[0]; 
		bytesToSwapArr[0] ^= binLocus[iByte];
		binLocus[iByte]   ^= bytesToSwapArr[0]; 
		memcpy(&bytesToSwap, bytesToSwapArr.data(), 1);
		// Now expand the bits swapped from the current binLocus byte to the mask positions in the aggregate word (Chapter 7-5 of Hacker's Delight)
		std::array<uint64_t, 6> a;
		m  = swapBitMask;
		mk = ~m << 1;
		for (uint32_t i = 0; i < 6; ++i){
			mp   = mk ^ (mk << 1);
			mp   = mp ^ (mp << 2);
			mp   = mp ^ (mp << 4);
			mp   = mp ^ (mp << 8);
			mp   = mp ^ (mp << 16);
			mp   = mp ^ (mp << 32);
			mv   = mp & m;
			a[i] = mv;
			m    = (m ^ mv) | ( mv >> (1 << i) );
			mk   = mk & ~mp;
		}
		bytesToSwap  = (bytesToSwap & ~a[5]) | ( (bytesToSwap << 32) & a[5] );
		bytesToSwap  = (bytesToSwap & ~a[4]) | ( (bytesToSwap << 16) & a[4] );
		bytesToSwap  = (bytesToSwap & ~a[3]) | ( (bytesToSwap << 8) & a[3] );
		bytesToSwap  = (bytesToSwap & ~a[2]) | ( (bytesToSwap << 4) & a[2] );
		bytesToSwap  = (bytesToSwap & ~a[1]) | ( (bytesToSwap << 2) & a[1] );
		bytesToSwap  = (bytesToSwap & ~a[0]) | ( (bytesToSwap << 1) & a[0] );
		bytesToSwap &= swapBitMask;
		memcpy(bytesToSwapArr.data(), &bytesToSwap, byteSize);
		// Finally, replace the relevant bits in the binLocus bytes indexed by the permutation
		// Order of operations is important!
		// Chapter 2-20 of Hacker's Delight, but we do not need the full swap
		for (size_t inByteInd = 0; inByteInd < byteSize; ++inByteInd){
			binLocus[ permByteInd[inByteInd] ] ^= (binLocus[ permByteInd[inByteInd] ] ^ bytesToSwapArr[inByteInd]) & swapBitMaskArr[inByteInd];
		}
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	permTime   = time2 - time1;
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
	//const size_t nIndividuals    = 125;
	const size_t nIndividuals    = 129;
	//const size_t kSketches       = 100;
	const size_t kSketches       = 20;
	const size_t locusSize       = nIndividuals / 8 + static_cast<bool>(nIndividuals % 8);
	const size_t sketchSize      = nIndividuals / kSketches + static_cast<bool>(nIndividuals % kSketches);
	const uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	const size_t ranBitVecSize   = nIndividuals / (sizeof(uint64_t) * 8) + static_cast<bool> ( nIndividuals % (sizeof(uint64_t) * 8) );
	std::vector<uint32_t> seeds{static_cast<uint32_t>( prng.ranInt() )};
	std::vector<size_t> ranIntsUp{prng.fyIndexesUp(nIndividuals)};
	std::vector<uint16_t> sketches(kSketches, emptyBinToken);
	std::vector<uint8_t> binLocus;
	std::vector<uint64_t> ranBits;
	for (size_t i = 0; i < ranBitVecSize; ++i){
		ranBits.push_back( prng.ranInt() );
	}
	for (const auto riu : ranIntsUp){
		std::cout << riu << " ";
	}
	std::cout << "\n";
	std::bitset<ranBitVecSize * 64> ranBitBS{0};
	for (size_t j = 0; j < ranBitVecSize; ++j){
		for (size_t i = 0; i < 64; ++i){
			const size_t ij = i + j * 64;
			ranBitBS[ij] = (ranBits[j] >> ij) & static_cast<uint64_t>(1);
		}
	}
	std::cout << ranBitBS << "\n";
	for (size_t i = 0; i < nIndividuals - 1; ++i){
		bool tmp = ranBitBS[i];
		ranBitBS[i]              = ranBitBS[ ranIntsUp[i] ];
		ranBitBS[ ranIntsUp[i] ] = tmp;
	}
	std::cout << ranBitBS << "\n";
	//std::cout << std::bitset<64>(ranBits[1]) << std::bitset<64>(ranBits[0]) << "\n";
	for (const auto rb : ranBits){
		const auto bits = reinterpret_cast<const uint8_t *>(&rb);
		for (size_t ii = 0; ii < sizeof(uint64_t); ++ii){
			binLocus.push_back(bits[ii]);
		}
	}
	/*
	for (auto blIt = binLocus.rbegin(); blIt != binLocus.rend(); ++blIt){
		std::cout << std::bitset<8>(*blIt);
	}
	std::cout << "\n";
	*/
	//binLocus2.back() = binLocus2.back() >> 3;
	//uint64_t xi = _pext_u64(ranBits[0], m); Compress
	//x = _pdep_u64(xi, mEx); Expand
	const std::array<float, 2> res2 = locusOPH(0, nIndividuals, locusSize, kSketches, sketchSize, ranIntsUp, seeds, prng, binLocus, sketches);
	for (auto blIt = binLocus.rbegin(); blIt != binLocus.rend(); ++blIt){
		std::cout << std::bitset<8>(*blIt);
	}
	std::cout << "\n";
	/*
	std::cout << res1[0] << "\t" << res1[1] << "\t" << res2[0] << "\t" << res2[1] << "\n";
	*/
}

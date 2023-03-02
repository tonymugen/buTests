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

/** \file Testing stand-alone MurMurHash3 implementations */

#include <vector>
#include <array>
#include <cstring>
#include <iostream>

#include "random.hpp"

uint32_t murMurHashOld(const size_t &startInd, const size_t &nElements, std::vector<uint16_t> &sketches, const uint32_t &seed) {
	// TODO: add an assert() on nElements != 0 and startInd < sketches_.size()
	uint32_t hash  = seed;
	auto blocks    = reinterpret_cast<const uint32_t *>(sketches.data() + startInd);
	size_t nBlocks = nElements / 2; // each sketch is 16 bits

	// body
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock){
		uint32_t k1 = blocks[iBlock];

		k1 *= 0xcc9e2d51;
		k1  = (k1 << 15) | (k1 >> 17);
		k1 *= 0x1b873593;

		hash ^= k1;
		hash  = (hash << 13) | (hash >> 19);
		hash  = hash * 5 + 0xe6546b64;
	}

	// tail, if exists
	if (nElements % 2){
		uint32_t k1 = static_cast<uint32_t>(sketches[startInd + nElements - 1]);

		k1 *= 0xcc9e2d51;
		k1  = (k1 << 15) | (k1 >> 17);
		k1 *= 0x1b873593;

		hash ^= k1;
		hash  = (hash << 13) | (hash >> 19);
		hash  = hash * 5 + 0xe6546b64;
	}

	// finalize
	hash ^= sizeof(size_t);
	hash ^= hash >> 16;
	hash *= 0x85ebca6b;
	hash ^= hash >> 13;
	hash *= 0xc2b2ae35;
	hash ^= hash >> 16;

	return hash;
}

uint32_t murMurHashOld(const size_t &key, const uint32_t &seed) {
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

uint32_t murMurHashMixer(const size_t &key, const uint32_t &seed){
	constexpr uint32_t const1{0xcc9e2d51};
	constexpr uint32_t const2{0x1b873593};
	constexpr size_t   nBlocks32{sizeof(size_t) / sizeof(uint32_t)};    // number of 32 bit blocks in size_t
	constexpr uint32_t keyLen{sizeof(size_t)};                          // key length 
	constexpr uint32_t hashMultiplier{5};
	constexpr uint32_t hashAdder{0xe6546b64};

	constexpr std::array<uint32_t, 4> blockShifts{15, 17, 13, 19};

	uint32_t hash{seed};
	std::array<uint32_t, nBlocks32> blocks{0};
	memcpy(blocks.data(), &key, keyLen);

	// body
	for (auto &eachBlock : blocks){
		eachBlock *= const1;
		eachBlock  = (eachBlock << blockShifts[0]) | (eachBlock >> blockShifts[1]);
		eachBlock *= const2;

		hash ^= eachBlock;
		hash  = (hash << blockShifts[2]) | (hash >> blockShifts[3]);
		hash  = hash * hashMultiplier + hashAdder;
	}
	return hash;
}

uint32_t murMurHashFinalizer(const uint32_t &hashIn){
	constexpr uint32_t keyLen{sizeof(size_t)};                          // key length 
	constexpr std::array<uint32_t, 2> finalizeShifts{16, 13};
	constexpr std::array<uint32_t, 2> finalizeMult{0x85ebca6b, 0xc2b2ae35};

	uint32_t hash = hashIn;
	hash ^= keyLen;
	hash ^= hash >> finalizeShifts[0];
	hash *= finalizeMult[0];
	hash ^= hash >> finalizeShifts[1];
	hash *= finalizeMult[1];
	hash ^= hash >> finalizeShifts[0];

	return hash;
}

uint32_t murMurHash(const size_t &key, const uint32_t &seed){
	constexpr uint32_t const1{0xcc9e2d51};
	constexpr uint32_t const2{0x1b873593};
	constexpr size_t   nBlocks32{sizeof(size_t) / sizeof(uint32_t)};    // number of 32 bit blocks in size_t
	constexpr uint32_t keyLen{sizeof(size_t)};                          // key length 
	constexpr uint32_t hashMultiplier{5};
	constexpr uint32_t hashAdder{0xe6546b64};

	constexpr std::array<uint32_t, 4> blockShifts{15, 17, 13, 19};
	constexpr std::array<uint32_t, 2> finalizeShifts{16, 13};
	constexpr std::array<uint32_t, 2> finalizeMult{0x85ebca6b, 0xc2b2ae35};

	uint32_t hash{seed};
	std::array<uint32_t, nBlocks32> blocks{0};
	memcpy(blocks.data(), &key, keyLen);

	// body
	for (auto &eachBlock : blocks){
		eachBlock *= const1;
		eachBlock  = (eachBlock << blockShifts[0]) | (eachBlock >> blockShifts[1]);
		eachBlock *= const2;

		hash ^= eachBlock;
		hash  = (hash << blockShifts[2]) | (hash >> blockShifts[3]);
		hash  = hash * hashMultiplier + hashAdder;
	}

	// there is no tail since the input is fixed (at eight bytes typically)
	// finalize
	hash ^= keyLen;
	hash ^= hash >> finalizeShifts[0];
	hash *= finalizeMult[0];
	hash ^= hash >> finalizeShifts[1];
	hash *= finalizeMult[1];
	hash ^= hash >> finalizeShifts[0];

	return hash;
}

uint32_t murMurHash(const std::vector<size_t> &key, const uint32_t &seed) {
	uint32_t hash{seed};
	for (const auto &eachIdx : key){
		hash = murMurHashMixer(eachIdx, hash);
	}
	hash = murMurHashFinalizer(hash);
	return hash;
}

uint32_t murMurHash(const size_t &start, const size_t &length, const std::vector<uint16_t> &key, const uint32_t &seed) {
	constexpr size_t keysPerWord{sizeof(size_t) / sizeof(uint16_t)};
	constexpr size_t bytesPerWord{sizeof(size_t)};
	constexpr auto roundMask = static_cast<size_t>( -(keysPerWord) );
	uint32_t hash{seed};
	//const size_t end{start + length};
	//const size_t wholeEnd{end & roundMask};
	const size_t wholeLength{length & roundMask};
	const size_t wholeEnd{start + wholeLength};

	size_t keyIdx{start};
	while (keyIdx < wholeEnd){
		size_t keyBlock{0};
		memcpy(&keyBlock, key.data() + keyIdx, bytesPerWord);
		hash    = murMurHashMixer(keyBlock, hash);
		keyIdx += keysPerWord;
	}
	if (length > wholeLength){  // if there is a tail
		size_t keyBlock{0};
		memcpy(&keyBlock, key.data() + keyIdx, bytesPerWord);
		hash = murMurHashMixer(keyBlock, hash);
	}
	hash = murMurHashFinalizer(hash);
	return hash;
}

int main() {
	constexpr size_t numTests{100};
	constexpr size_t keyVecLen{233};
	constexpr size_t start{3};
	constexpr size_t length{151};
	BayesicSpace::RanDraw prng;
	uint32_t numberSame{0};
	for (size_t iStep = 0; iStep < numTests; ++iStep) {
		const auto seed  = static_cast<uint32_t>( prng.ranInt() );
		std::vector<uint16_t> key(keyVecLen, 0);
		for (auto &eachKey : key){
			eachKey = static_cast<uint16_t>( prng.ranInt() );
		}
		const uint32_t oldHash = murMurHashOld(start, length, key, seed);
		const uint32_t newHash = murMurHash(start, length, key, seed);
		numberSame += static_cast<uint32_t>(oldHash == newHash);
	}
	std::cout << "\nnumber same: " << numberSame << "\n";
}


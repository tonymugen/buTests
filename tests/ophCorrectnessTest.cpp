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

/** \file Check new OPH methods against the naive implementation
 *
 * This file runs checks on my new OPH derivation methods, comparing them to the previous implementation that does a simple sequential search implemented and tested previously.
 *
 */

#include <cstdint>
#include <vector>
#include <iostream>
#include <cstring>
#include <limits>
#include <array>
#include <immintrin.h>
#include <x86intrin.h>

#include "random.hpp"

uint32_t murMurHash(const size_t &key, const uint32_t &seed) {
	uint32_t hash = seed;
	constexpr std::array<uint32_t, 5> multConsts{0xcc9e2d51, 0x1b873593, 0xe6546b64, 0x85ebca6b, 0xc2b2ae35};
	constexpr std::array<uint32_t, 8> shiftConsts{15, 17, 13, 19, 5, 16, 13, 16};
	// body
	// TODO: change to memcpy; possible strict aliasing violation
	auto blocks = reinterpret_cast<const uint32_t *>(&key);

	for (size_t i = 0; i < sizeof(size_t) / sizeof(uint32_t); ++i){
		uint32_t k1 = blocks[i];

		k1 *= multConsts[0];
		k1  = (k1 << shiftConsts[0]) | (k1 >> shiftConsts[1]);
		k1 *= multConsts[1];

		hash ^= k1;
		hash  = (hash << shiftConsts[2]) | (hash >> shiftConsts[3]);
		hash  = hash * shiftConsts[4] + multConsts[2];
	}

	// there is no tail since the input is fixed (at eight bytes typically)
	// finalize
	hash ^= sizeof(size_t);
	hash ^= hash >> shiftConsts[5];
	hash *= multConsts[3];
	hash ^= hash >> shiftConsts[6];
	hash *= multConsts[4];
	hash ^= hash >> shiftConsts[7];

	return hash;
}

std::array<int32_t, 2> locusOPH(const size_t &nIndividuals, const size_t &kSketches, const uint64_t &seed, std::vector<uint8_t> &binLocus) {
	std::array<int32_t, 2> result{0, -1};

	BayesicSpace::RanDraw prng(seed);
	BayesicSpace::RanDraw prngNaive(seed);
	BayesicSpace::RanDraw prngMem(seed);
	BayesicSpace::RanDraw prngSmall(seed);
	std::vector<size_t> permutation{prng.fyIndexesUp(nIndividuals)};
	constexpr uint8_t byteSize        = 8;
	constexpr uint16_t oneBit         = 1;
	constexpr uint64_t wordSizeInBits = 64;
	constexpr size_t wordSize         = 8;
	const uint16_t emptyBinToken      = std::numeric_limits<uint16_t>::max();
	const size_t nFullBytes           = (nIndividuals - 1) / byteSize;
	const size_t sketchSize           = nIndividuals / kSketches;
	size_t iIndiv{0};
	size_t iByte{0};
	while(iByte < nFullBytes){
		for (uint8_t iInLocusByte = 0; iInLocusByte < byteSize; ++iInLocusByte){
			auto bytePair            = static_cast<uint16_t>(binLocus[iByte]);
			const size_t perIndiv    = permutation[iIndiv++];                                   // post-increment to use current value for index first
			const size_t permByteInd = perIndiv / byteSize;
			const auto permInByteInd = static_cast<uint8_t>(perIndiv % byteSize);
			// Pair the current locus byte with the byte containing the value to be swapped
			// Then use the exchanging two fields trick from Hacker's Delight Chapter 2-20
			bytePair                    |= static_cast<uint16_t>(binLocus[permByteInd]) << byteSize;
			const auto mask              = static_cast<uint16_t>(oneBit << iInLocusByte);
			const auto perMask           = static_cast<uint8_t>(oneBit << permInByteInd);
			const uint16_t shiftDistance = (byteSize - iInLocusByte) + permInByteInd;           // subtraction is safe b/c byteSize is the loop terminator
			const uint16_t temp1         = ( bytePair ^ (bytePair >> shiftDistance) ) & mask;
			const auto temp2             = static_cast<uint16_t>(temp1 << shiftDistance);
			bytePair                    ^= temp1 ^ temp2;
			// Transfer bits using the trick in Hacker's Delight Chapter 2-20 (do not need the full swap, just transfer from the byte pair to binLocus)
			// Must modify the current byte in each loop iteration because permutation indexes may fall into it
			binLocus[iByte]             ^= ( binLocus[iByte] ^ static_cast<uint8_t>(bytePair) ) & static_cast<uint8_t>(mask);
			binLocus[permByteInd]       ^= ( binLocus[permByteInd] ^ static_cast<uint8_t>(bytePair >> byteSize) ) & perMask;
 		}
		++iByte;
	}
	// Finish the individuals in the remaining partial byte, if any
	uint8_t iInLocusByte = 0;
	while (iIndiv < nIndividuals - 1){
		auto bytePair            = static_cast<uint16_t>(binLocus[iByte]);
		const size_t perIndiv    = permutation[iIndiv++];                                   // post-increment to use current value for index first
		const size_t permByteInd = perIndiv / byteSize;
		const auto permInByteInd = static_cast<uint8_t>(perIndiv % byteSize);
		// Pair the current locus byte with the byte containing the value to be swapped
		// Then use the exchanging two fields trick from Hacker's Delight Chapter 2-20
		bytePair                    |= static_cast<uint16_t>(binLocus[permByteInd]) << byteSize;
		const auto mask              = static_cast<uint16_t>(oneBit << iInLocusByte);
		const auto perMask           = static_cast<uint8_t>(oneBit << permInByteInd);
		const uint16_t shiftDistance = (byteSize - iInLocusByte) + permInByteInd;           // subtraction is safe b/c byteSize is the loop terminator
		const uint16_t temp1         = ( bytePair ^ (bytePair >> shiftDistance) ) & mask;
		const auto temp2             = static_cast<uint16_t>(temp1 << shiftDistance);
		bytePair                    ^= temp1 ^ temp2;
		// Transfer bits using the trick in Hacker's Delight Chapter 2-20 (do not need the full swap, just transfer from the byte pair to binLocus)
		// Must modify the current byte in each loop iteration because permutation indexes may fall into it
		binLocus[iByte]             ^= ( binLocus[iByte] ^ static_cast<uint8_t>(bytePair) ) & static_cast<uint8_t>(mask);
		binLocus[permByteInd]       ^= ( binLocus[permByteInd] ^ static_cast<uint8_t>(bytePair >> byteSize) ) & perMask;
		++iInLocusByte;
	}
	std::vector<size_t> filledIndexes;                // indexes of the non-empty sketches
	std::vector<uint16_t> sketchesNaive(kSketches, emptyBinToken);
	std::vector<uint32_t> seedsNaive{static_cast<uint32_t>( prngNaive.ranInt() )};
	iByte           = 0;
	size_t iSeed    = 0;                             // index into the seed vector
	uint8_t iInByte = 0;
	for (size_t iSketch = 0; iSketch < kSketches; ++iSketch){
		uint16_t firstSetBitPos = 0;
		while ( ( iByte != binLocus.size() ) && ( ( (oneBit << iInByte) & binLocus[iByte] ) == 0 ) &&
				(firstSetBitPos < sketchSize) ){
			++iInByte;
			// these are instead of an if statement
			iByte  += static_cast<size_t>(iInByte == byteSize);
			iInByte = iInByte % byteSize;
			++firstSetBitPos;
		}
		if ( ( iByte < binLocus.size() ) && (firstSetBitPos < sketchSize) ){
			filledIndexes.push_back(iSketch);
			sketchesNaive[iSketch]   = firstSetBitPos;
			const auto remainder     = static_cast<uint16_t>(sketchSize - firstSetBitPos);
			const uint16_t inByteMod = remainder % byteSize;
			const uint16_t inByteSum = iInByte + inByteMod;

			iByte  += remainder / byteSize + inByteSum / byteSize;
			iInByte = inByteSum % byteSize;
		}
	}
	if (filledIndexes.size() == 1){
		for (size_t iSk = 0; iSk < kSketches; ++iSk){ // this will overwrite the one assigned sketch, but the wasted operation should be swamped by the rest
			// should be safe: each thread accesses different vector elements
			sketchesNaive[iSk] = sketchesNaive[ filledIndexes[0] ];
		}
	} else if (filledIndexes.size() != kSketches){
		if ( filledIndexes.empty() ){ // in the case where the whole locus is monomorphic, pick a random index as filled
			filledIndexes.push_back( prngNaive.sampleInt(kSketches) );
		}
		size_t emptyCount = kSketches - filledIndexes.size();
		while (emptyCount > 0){
			for (const auto &fInd : filledIndexes){
				uint32_t newIdx = murMurHash(fInd, seedsNaive[iSeed]) % kSketches;
				// should be safe: each thread accesses different vector elements
				if (sketchesNaive[newIdx] == emptyBinToken){
					sketchesNaive[newIdx] = sketchesNaive[fInd];
					--emptyCount;
					break;
				}
			}
			++iSeed;
			if ( iSeed == seedsNaive.size() ){
				seedsNaive.push_back( static_cast<uint32_t>( prngNaive.ranInt() ) );
			}
		}
	}

	filledIndexes.clear();
	std::vector<uint16_t> sketchesMem(kSketches, emptyBinToken);
	std::vector<uint32_t> seedsMem{static_cast<uint32_t>( prngMem.ranInt() )};
	iByte = 0;
	constexpr uint64_t allBitsSet{std::numeric_limits<uint64_t>::max()};
	size_t iSketch{0};
	uint64_t sketchTail{0};                                                            // left over buts from beyond the last full byte of the previous sketch
	size_t locusChunkSize = (wordSize > binLocus.size() ? binLocus.size() : wordSize);
	while ( iByte < binLocus.size() ){
		uint64_t nWordUnsetBits{wordSizeInBits};
		uint64_t nSumUnsetBits{0};
		while ( (nWordUnsetBits == wordSizeInBits) && ( iByte < binLocus.size() ) ){
			uint64_t locusChunk{allBitsSet};
			// TODO: add cassert() for iByte < nBytesToHash; must be true since this is the loop conditions
			const size_t nRemainingBytes{binLocus.size() - iByte};
			locusChunkSize = static_cast<size_t>(nRemainingBytes >= wordSize) * wordSize + static_cast<size_t>(nRemainingBytes < wordSize) * nRemainingBytes;
			memcpy(&locusChunk, binLocus.data() + iByte, locusChunkSize);
			locusChunk    &= allBitsSet << sketchTail;
			nWordUnsetBits = _tzcnt_u64(locusChunk);
			nSumUnsetBits += nWordUnsetBits - sketchTail;
			sketchTail     = 0;
			iByte         += locusChunkSize;
		}
		iSketch += nSumUnsetBits / sketchSize;
		if (iSketch >= kSketches){
			break;
		}
		filledIndexes.push_back(iSketch);
		sketchesMem[iSketch] = static_cast<uint16_t>(nSumUnsetBits % sketchSize);
		++iSketch;
		const uint64_t bitsDone{iSketch * sketchSize};
		iByte      = bitsDone / byteSize;
		sketchTail = bitsDone % byteSize;
	}
	// Must deal with the potential overshoot in sketch number if the last chunk of sketches is empty
	if ( (!filledIndexes.empty() ) && (filledIndexes.back() >= kSketches) ){
		filledIndexes.pop_back();
	}
	iSeed = 0;
	if (filledIndexes.size() == 1){
		for (size_t iSk = 0; iSk < kSketches; ++iSk){ // this will overwrite the one assigned sketch, but the wasted operation should be swamped by the rest
			// should be safe: each thread accesses different vector elements
			sketchesMem[iSk] = sketchesMem[ filledIndexes[0] ];
		}
	} else if (filledIndexes.size() != kSketches){
		if ( filledIndexes.empty() ){ // in the case where the whole locus is monomorphic, pick a random index as filled
			filledIndexes.push_back( prngMem.sampleInt(kSketches) );
		}
		size_t emptyCount = kSketches - filledIndexes.size();
		while (emptyCount > 0){
			for (const auto &fInd : filledIndexes){
				uint32_t newIdx = murMurHash(fInd, seedsMem[iSeed]) % kSketches;
				// should be safe: each thread accesses different vector elements
				if (sketchesMem[newIdx] == emptyBinToken){
					sketchesMem[newIdx] = sketchesMem[fInd];
					--emptyCount;
					break;
				}
			}
			++iSeed;
			if ( iSeed == seedsMem.size() ){
				seedsMem.push_back( static_cast<uint32_t>( prngMem.ranInt() ) );
			}
		}
	}
	for (size_t jSketch = 0; jSketch < kSketches; ++jSketch){
		result[0] += static_cast<int32_t>(sketchesNaive[jSketch] != sketchesMem[jSketch]);
	}
	filledIndexes.clear();
	if (sketchSize < wordSizeInBits){
		std::vector<uint16_t> sketchesSmall(kSketches, emptyBinToken);
		std::vector<uint32_t> seedsSmall{static_cast<uint32_t>( prngSmall.ranInt() )};
		iSketch = 0;
		iByte   = 0;
		constexpr uint64_t roundMask{0xfffffffffffffff8};
		const size_t nWholeWordBytes{binLocus.size() & roundMask};
		uint64_t initialShift{0};
		uint64_t carryOverZeros{0};
		while (iByte < nWholeWordBytes){
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, wordSize);
			uint64_t trackingMask{std::numeric_limits<uint64_t>::max()};
			if (carryOverZeros != 0){
				uint64_t splitSketch{locusChunk | (trackingMask << initialShift)};
				splitSketch = splitSketch << carryOverZeros;
				const uint64_t lastSketch{_tzcnt_u64(splitSketch)};
				if (lastSketch < sketchSize){
					const size_t prevIsketch{iSketch - 1};
					if (prevIsketch >= kSketches){
						break;
					}
					filledIndexes.push_back(prevIsketch);
					sketchesSmall[prevIsketch] = static_cast<uint16_t>(lastSketch);
				}
			}
			locusChunk   = locusChunk >> initialShift;
			trackingMask = trackingMask >> initialShift;
			uint64_t iShift{0};
			uint64_t setBit{0};
			while (locusChunk != 0){
				setBit = _tzcnt_u64(locusChunk);                           // trailing zero count counts from the correct end
				const uint64_t idxToSkip{setBit / sketchSize};
				iShift       = sketchSize * (idxToSkip + 1);
				locusChunk   = locusChunk >> iShift;
				trackingMask = trackingMask >> iShift;
				// if iShift > 63, the shift result is undefined, but we want all 0
				const auto finalFix = static_cast<uint64_t>( -static_cast<uint64_t>(iShift < wordSizeInBits) );
				locusChunk         &= finalFix;
				trackingMask       &= finalFix;
				iSketch            += idxToSkip;
				if (iSketch >= kSketches){
					break;
				}
				filledIndexes.push_back(iSketch);
				sketchesSmall[iSketch] = static_cast<uint16_t>(setBit % sketchSize);             // should be safe: each thread accesses different vector elements
				++iSketch;
			}
			const uint64_t trailingSetBit{_tzcnt_u64(~trackingMask)};
			const uint64_t trailingWholeSketches{trailingSetBit / sketchSize};
			carryOverZeros  = trailingSetBit % sketchSize;
			iSketch        += trailingWholeSketches + static_cast<uint64_t>(carryOverZeros > 0);
			initialShift    = (sketchSize * iSketch) % wordSizeInBits;
			iByte          += wordSize;
		}
		if ( iByte < binLocus.size() ){
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, binLocus.size() - iByte);              // subtraction is safe inside the iByte < nBytesToHash test
			if (carryOverZeros != 0){
				constexpr uint64_t trackingMask{std::numeric_limits<uint64_t>::max()};
				uint64_t splitSketch{locusChunk | (trackingMask << initialShift)};
				splitSketch = splitSketch << carryOverZeros;
				const uint64_t lastSketch{_tzcnt_u64(splitSketch)};
				if (lastSketch < sketchSize){
					const size_t prevIsketch{iSketch - 1};
					filledIndexes.push_back(prevIsketch);
					sketchesSmall[prevIsketch] = static_cast<uint16_t>(lastSketch);
				}
			}
			locusChunk   = locusChunk >> initialShift;
			uint64_t iShift{0};
			while (locusChunk != 0){
				const uint64_t setBit{_tzcnt_u64(locusChunk)};                            // trailing zero count counts from the correct end
				const uint64_t idxToSkip{setBit / sketchSize};
				iShift     = sketchSize * (idxToSkip + 1);
				locusChunk = locusChunk >> iShift;
				// if iShift > 63, the shift result is undefined, but we want all 0
				const auto finalFix = static_cast<uint64_t>( -static_cast<uint64_t>(iShift < wordSizeInBits) );
				locusChunk         &= finalFix;
				iSketch            += idxToSkip;
				if (iSketch >= kSketches){
					break;
				}
				filledIndexes.push_back(iSketch);
				sketchesSmall[iSketch] = static_cast<uint16_t>(setBit % sketchSize);             // should be safe: each thread accesses different vector elements
				++iSketch;
			}
		}
		if ( (!filledIndexes.empty() ) && (filledIndexes.back() >= kSketches) ){
			filledIndexes.pop_back();
		}
		iSeed = 0;
		if (filledIndexes.size() == 1){
			for (size_t iSk = 0; iSk < kSketches; ++iSk){ // this will overwrite the one assigned sketch, but the wasted operation should be swamped by the rest
				// should be safe: each thread accesses different vector elements
				sketchesSmall[iSk] = sketchesSmall[ filledIndexes[0] ];
			}
		} else if (filledIndexes.size() != kSketches){
			if ( filledIndexes.empty() ){ // in the case where the whole locus is monomorphic, pick a random index as filled
				filledIndexes.push_back( prngSmall.sampleInt(kSketches) );
			}
			size_t emptyCount = kSketches - filledIndexes.size();
			while (emptyCount > 0){
				for (const auto &fInd : filledIndexes){
					uint32_t newIdx = murMurHash(fInd, seedsSmall[iSeed]) % kSketches;
					// should be safe: each thread accesses different vector elements
					if (sketchesSmall[newIdx] == emptyBinToken){
						sketchesSmall[newIdx] = sketchesSmall[fInd];
						--emptyCount;
						break;
					}
				}
				++iSeed;
				if ( iSeed == seedsSmall.size() ){
					seedsSmall.push_back( static_cast<uint32_t>( prngSmall.ranInt() ) );
				}
			}
		}
		result[1] = 0;
		for (size_t jSketch = 0; jSketch < kSketches; ++jSketch){
			result[1] += static_cast<int32_t>(sketchesNaive[jSketch] != sketchesSmall[jSketch]);
		}
	}
	return result;
}

int main(){
	constexpr size_t minKsketches   = 3;
	constexpr size_t maxKsketches   = 100;
	constexpr size_t maxIndMult     = 100;
	constexpr uint64_t roundMask    = 0xfffffffffffffff8;
	constexpr size_t wordSizeInBits = 64;
	constexpr size_t wordSize       = 8;
	constexpr size_t byteSize       = 8;
	std::cout << "nIndividuals\tnIndivToHash\tkSketches\tseed\tMEMsum\tSmallSum\n";
	for (size_t kSketches = minKsketches; kSketches <= maxKsketches; ++kSketches){
		for (size_t nIndividuals = kSketches * 2; nIndividuals <= kSketches * maxIndMult; ++nIndividuals){
			BayesicSpace::RanDraw seedPRNG;
			const uint64_t seedInt{seedPRNG.ranInt()};
			BayesicSpace::RanDraw prng1(seedInt);
			const size_t sketchSize    = nIndividuals / kSketches + static_cast<size_t>( (nIndividuals % kSketches) > 0 );
			const size_t nIndivToHash  = sketchSize * kSketches;                                                   // round up to the nearest divisible by kSketches number
			const size_t locusSize     = ( ( nIndivToHash + (byteSize - 1) ) & roundMask ) / byteSize;             // round up the nIndivToHash to the nearest multiple of 8
			const size_t ranBitVecSize = nIndivToHash / wordSizeInBits + static_cast<size_t>( (nIndivToHash % wordSizeInBits) > 0 );
			std::vector<uint8_t> binLocus;
			/*
			std::vector<uint64_t> ranBits;
			for (size_t i = 0; i < ranBitVecSize; ++i){
				ranBits.push_back( seedPRNG.ranInt() );
			}
			size_t iByte = 0;
			*/
			binLocus.resize(locusSize, 0);
			binLocus[0] = 0b00001111;
			/*
			for (const auto eachWord : ranBits){
				for (size_t byteInWord = 0; (byteInWord < wordSize) && (iByte < locusSize); ++byteInWord){
					//binLocus[iByte] = static_cast<uint8_t>( eachWord >> (byteInWord * byteSize) ) & (static_cast<uint8_t>(1) << byteInWord);
					binLocus[iByte] = static_cast<uint8_t>( eachWord >> (byteInWord * byteSize) );
					++iByte;
				}
			}
			*/
			// pad out the extra individuals by randomly sampling from the given set
			for (size_t iAddIndiv = nIndividuals; iAddIndiv < nIndivToHash; ++iAddIndiv){
				const size_t iLocByte    = iAddIndiv / byteSize;
				const auto iInLocByte    = static_cast<uint8_t>(iAddIndiv % byteSize);
				auto bytePair            = static_cast<uint16_t>(binLocus[iLocByte]);
				const size_t perIndiv    = seedPRNG.sampleInt(nIndividuals);
				const size_t permByteInd = perIndiv / byteSize;
				const auto permInByteInd = static_cast<uint8_t>(perIndiv % byteSize);
				// Pair the current locus byte with the byte containing the value to be swapped
				// Then use the exchanging two fields trick from Hacker's Delight Chapter 2-20
				bytePair                    |= static_cast<uint16_t>(binLocus[permByteInd]) << byteSize;
				const auto mask              = static_cast<uint16_t>(1 << iInLocByte);
				const uint16_t shiftDistance = (byteSize - iInLocByte) + permInByteInd;                        // subtraction is safe b/c iInLocByte is modulo byteSize
				const uint16_t temp1         = ( bytePair ^ (bytePair >> shiftDistance) ) & mask;
				const auto temp2             = static_cast<uint16_t>(temp1 << shiftDistance);
				bytePair                    ^= temp1 ^ temp2;
				// Transfer bits using the trick in Hacker's Delight Chapter 2-20 (do not need the full swap, just transfer from the byte pair to binLocus1)
				// Must modify the current byte in each loop iteration because permutation indexes may fall into it
				binLocus[iLocByte]         ^= ( binLocus[iLocByte] ^ static_cast<uint8_t>(bytePair) ) & static_cast<uint8_t>(mask);
			}
			if ( (nIndivToHash % byteSize) > 0 ){
				binLocus.back() |= std::numeric_limits<uint8_t>::max() << static_cast<uint8_t>(nIndivToHash % byteSize);
			}
			std::cout << nIndividuals << "\t" << nIndivToHash << "\t" << kSketches << "\t" << seedInt << "\t";
			const std::array<int32_t, 2> res = locusOPH(nIndivToHash, kSketches, seedInt, binLocus);
			std::cout << res[0] << "\t" << res[1] << "\n";
		}
	}
}


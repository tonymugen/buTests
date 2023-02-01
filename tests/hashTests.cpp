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
#include <iomanip>
#include <chrono>
#include <cstring>
#include <limits>
#include <array>
#include <bitset>
#include <immintrin.h>
#include <x86intrin.h>

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

std::array<float, 2> locusOPHold(const size_t &locusInd, const size_t &nIndividuals, const size_t &locusSize, const size_t &kSketches, const size_t &sketchSize,
	const std::vector<size_t> &permutation, std::vector<uint32_t> &seeds, BayesicSpace::RanDraw &rng, std::vector<uint8_t> &binLocus, std::vector<uint16_t> &sketches) {
	std::chrono::duration<float, std::milli> permTime{0};
	std::chrono::duration<float, std::milli> sketchTime{0};
	// Start with a permutation to make OPH
	auto time1                       = std::chrono::high_resolution_clock::now();
	constexpr uint8_t byteSize       = 8;
	constexpr uint16_t oneBit        = 1;
	constexpr uint64_t roundMask     = 0xfffffffffffffff8;
	const uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	// Calculate the number of full bytes; nIndividuals - 1 because Fisher-Yates goes up to N - 2 inclusive
	const size_t nFullBytes = (nIndividuals - 1) / byteSize;
	size_t iIndiv           = 0;
	size_t iByte            = 0;
	while(iByte < nFullBytes){
		for (uint8_t iInLocusByte = 0; iInLocusByte < byteSize; ++iInLocusByte){
			auto bytePair            = static_cast<uint16_t>(binLocus[iByte]);
			const size_t perIndiv    = permutation[iIndiv++];                                   // post-increment to use current value for index first
			const size_t permByteInd = perIndiv / byteSize;
			const auto permInByteInd = static_cast<uint8_t>( perIndiv - (perIndiv & roundMask) );
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
		const auto permInByteInd = static_cast<uint8_t>( perIndiv - (perIndiv & roundMask) );
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
	auto time2 = std::chrono::high_resolution_clock::now();
	permTime   = time2 - time1;
	// Now make the sketches
	time1 = std::chrono::high_resolution_clock::now();
	std::vector<size_t> filledIndexes;                // indexes of the non-empty sketches
	iByte            = 0;
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
			iByte  += static_cast<size_t>(iInByte == byteSize);
			iInByte = iInByte % byteSize;
			++firstSetBitPos;
		}
		if ( (iByte < colEnd) && (firstSetBitPos < sketchSize) ){
			filledIndexes.push_back(iSketch);
			{
				// should be safe: each thread accesses different vector elements
				sketches[sketchBeg + iSketch] = firstSetBitPos;
			}

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
	sketchTime = time2 - time1;
	return std::array<float, 2>{permTime.count(), sketchTime.count()};
}

std::array<float, 2> locusOPH(const size_t &locusInd, const size_t &nIndividuals, const size_t &locusSize, const size_t &kSketches, const size_t &sketchSize,
	const std::vector<size_t> &permutation, std::vector<uint32_t> &seeds, BayesicSpace::RanDraw &rng, std::vector<uint8_t> &binLocus, std::vector<uint16_t> &sketches) {
	std::chrono::duration<float, std::milli> permTime{0};
	std::chrono::duration<float, std::milli> sketchTime{0};
	// Start with a permutation to make OPH
	auto time1                      = std::chrono::high_resolution_clock::now();
	constexpr uint8_t byteSize      = 8;
	constexpr size_t wordSize       = 8;
	constexpr uint16_t oneBit       = 1;
	constexpr uint64_t roundMask    = 0xfffffffffffffff8;
	constexpr size_t wordSizeInBits = 64;
	// Calculate the number of full bytes; nIndividuals - 1 because Fisher-Yates goes up to N - 2 inclusive
	const size_t nFullBytes = (nIndividuals - 1) / byteSize;
	size_t iIndiv           = 0;
	size_t iByte            = 0;
	while(iByte < nFullBytes){
		for (uint8_t iInLocusByte = 0; iInLocusByte < byteSize; ++iInLocusByte){
			auto bytePair            = static_cast<uint16_t>(binLocus[iByte]);
			const size_t perIndiv    = permutation[iIndiv++];                                   // post-increment to use current value for index first
			const size_t permByteInd = perIndiv / byteSize;
			const auto permInByteInd = static_cast<uint8_t>( perIndiv - (perIndiv & roundMask) );
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
		const auto permInByteInd = static_cast<uint8_t>( perIndiv - (perIndiv & roundMask) );
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
	auto time2 = std::chrono::high_resolution_clock::now();
	permTime   = time2 - time1;
	// Now make the sketches
	time1 = std::chrono::high_resolution_clock::now();
	constexpr uint16_t emptyBinToken{std::numeric_limits<uint16_t>::max()};
	const size_t nBytesToHash{(kSketches * sketchSize) / byteSize};
	const size_t tailBytesToHash{nBytesToHash % byteSize};
	std::cout << "nBytesToHash = " << nBytesToHash << "\n";
	std::vector<size_t> filledIndexes;                                                 // indexes of the non-empty sketches
	iByte = 0;
	size_t sketchBeg{locusInd * kSketches};
	constexpr uint64_t allBitsSet{std::numeric_limits<uint64_t>::max()};
	size_t iSketch{0};
	uint64_t sketchTail{0};                                                            // left over buts from beyond the last full byte of the previous sketch
	size_t locusChunkSize = (wordSize > nBytesToHash ? nBytesToHash : wordSize);
	while (iByte < nBytesToHash){
		uint64_t nWordUnsetBits{wordSizeInBits};
		uint64_t nSumUnsetBits{0};
		while ( (nWordUnsetBits == wordSizeInBits) && (iByte < nBytesToHash) ){
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, locusChunkSize);
			std::cout << std::bitset<wordSizeInBits>(locusChunk) << "\n";
			locusChunk     &= allBitsSet << sketchTail;
			std::cout << std::bitset<wordSizeInBits>(locusChunk) << "\n\n";
			nWordUnsetBits = _tzcnt_u64(locusChunk);
			nSumUnsetBits += nWordUnsetBits - sketchTail;
			sketchTail     = 0;
			// TODO: add cassert() for iByte < nBytesToHash; must be true since this is the loop conditions
			// the following two lines must be before the iByte increment, otherwise we are subtracting before the test and may wrap
			const size_t nRemainingWholeBytes{nBytesToHash - iByte};
			const size_t nRemainingWholeWords{nRemainingWholeBytes / wordSize};
			iByte         += locusChunkSize;
			locusChunkSize = static_cast<size_t>(nRemainingWholeWords > 0) * wordSize + static_cast<size_t>(nRemainingWholeWords == 0) * tailBytesToHash;
		}
		iSketch += nSumUnsetBits / sketchSize;
		std::cout << iSketch << "|" << nWordUnsetBits << "|" << nSumUnsetBits << "|" << nSumUnsetBits % sketchSize << "|" << locusChunkSize << "|";
		filledIndexes.push_back(iSketch);
		sketches[sketchBeg + iSketch] = static_cast<uint16_t>(nSumUnsetBits % sketchSize);
		++iSketch;
		const uint64_t bitsDone{iSketch * sketchSize};
		iByte      = bitsDone / byteSize;
		sketchTail = bitsDone % byteSize;
		std::cout << iByte << "|" << sketchTail << "\n";
	}
	for (const auto flInd : filledIndexes){
		std::cout << flInd << " ";
	}
	std::cout << "\n";
	/*
	if (sketchSize >= wordSizeInBits){
		for (auto blIt = binLocus.rbegin(); blIt != binLocus.rend(); ++blIt){
			std::cout << std::bitset<byteSize>(*blIt);
		}
		std::cout << "\n";
		constexpr uint64_t allBitsSet{std::numeric_limits<uint64_t>::max()};
		size_t iSketch{0};
		bool lastWordUnset{false};
		uint64_t wordPortion{wordSizeInBits};
		while (iByte < nEvenBytesToHash){
			std::cout << iSketch << "|" << static_cast<uint16_t>(lastWordUnset) << "|";
			const uint64_t initialShift = (sketchSize * iSketch) % wordSizeInBits;
			std::cout << initialShift << "|";
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, wordSize);
			if (lastWordUnset){
				const uint64_t lastSketchTail{locusChunk | (allBitsSet << initialShift)};
				const uint64_t setBit{_tzcnt_u64(lastSketchTail) + wordPortion};
				if (setBit < sketchSize){
					// TODO: add cassert() for iSketch > 0
					filledIndexes.push_back(iSketch);
					sketches[sketchBeg + iSketch] = static_cast<uint16_t>(setBit);
					++iSketch;
				}
			}
			locusChunk = locusChunk >> initialShift;
			const uint64_t trackingMask = ~(allBitsSet >> initialShift);
			wordPortion = _tzcnt_u64(trackingMask);
			const uint64_t setBit{_tzcnt_u64(locusChunk)};
			std::cout << setBit << " ";
			if (setBit < wordPortion){
				lastWordUnset = false;
				filledIndexes.push_back(iSketch);
				sketches[sketchBeg + iSketch] = static_cast<uint16_t>(setBit);
				++iSketch;
			} else {
				lastWordUnset = true;
			}
			iByte += wordSize;
		}
		std::cout << "\n";
		// do the last bytes if any (only if the beginning of the last hash is all 0)
		if ( (iByte < nBytesToHash) && lastWordUnset ){
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, nBytesToHash - iByte);              // subtraction is safe inside the iByte < nBytesToHash test
			const uint64_t setBit = _tzcnt_u64(locusChunk) + wordPortion;
			if (setBit < sketchSize){
				filledIndexes.push_back(iSketch);
				sketches[sketchBeg + iSketch] = static_cast<uint16_t>(setBit);
			}
		}
	} else {
		size_t iSketch{0};
		uint64_t initialShift{0};
		uint64_t carryOverZeros{0};
		while (iByte < nEvenBytesToHash){
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, wordSize);
			uint64_t trackingMask{std::numeric_limits<uint64_t>::max()};
			if (carryOverZeros != 0){
				uint64_t splitSketch{locusChunk | (trackingMask << initialShift)};
				splitSketch = splitSketch << carryOverZeros;
				const uint64_t lastSketch{_tzcnt_u64(splitSketch)};
				if (lastSketch < sketchSize){
					const size_t prevIsketch{iSketch - 1};
					filledIndexes.push_back(prevIsketch);
					sketches[sketchBeg + prevIsketch] = static_cast<uint16_t>(lastSketch);
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
				filledIndexes.push_back(iSketch);
				sketches[sketchBeg + iSketch] = static_cast<uint16_t>(setBit % sketchSize);             // should be safe: each thread accesses different vector elements
				++iSketch;
			}
			const uint64_t trailingSetBit{_tzcnt_u64(~trackingMask)};
			const uint64_t trailingWholeSketches{trailingSetBit / sketchSize};
			carryOverZeros  = trailingSetBit % sketchSize;
			iSketch        += trailingWholeSketches + static_cast<uint64_t>(carryOverZeros > 0);
			initialShift    = (sketchSize * iSketch) % wordSizeInBits;
			iByte          += wordSize;
		}
		if (iByte < nBytesToHash){
			uint64_t locusChunk{0};
			memcpy(&locusChunk, binLocus.data() + iByte, nBytesToHash - iByte);              // subtraction is safe inside the iByte < nBytesToHash test
			if (carryOverZeros != 0){
				constexpr uint64_t trackingMask{std::numeric_limits<uint64_t>::max()};
				uint64_t splitSketch{locusChunk | (trackingMask << initialShift)};
				splitSketch = splitSketch << carryOverZeros;
				const uint64_t lastSketch{_tzcnt_u64(splitSketch)};
				if (lastSketch < sketchSize){
					const size_t prevIsketch{iSketch - 1};
					filledIndexes.push_back(prevIsketch);
					sketches[sketchBeg + prevIsketch] = static_cast<uint16_t>(lastSketch);
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
				filledIndexes.push_back(iSketch);
				sketches[sketchBeg + iSketch] = static_cast<uint16_t>(setBit % sketchSize);             // should be safe: each thread accesses different vector elements
				++iSketch;
			}
		}
	}
	*/
	size_t iSeed{0};                                                                   // index into the seed vector
	if (filledIndexes.size() == 1){
		for (size_t iSk = 0; iSk < kSketches; ++iSk){ // this will overwrite the one assigned sketch, but the wasted operation should be swamped by the rest
			// should be safe: each thread accesses different vector elements
			sketches[sketchBeg + iSk] = sketches[filledIndexes[0] + sketchBeg];
		}
	} else if (filledIndexes.size() != kSketches){
		if ( filledIndexes.empty() ){ // in the case where the whole locus is monomorphic, pick a random index as filled
			filledIndexes.push_back( rng.sampleInt(kSketches) );
		}
		if (filledIndexes.size() > kSketches){
			throw std::string("more filled than K");
		}
		size_t emptyCount = kSketches - filledIndexes.size();
		while (emptyCount > 0){
			for (const auto &eachIdx : filledIndexes){
				size_t newIdx = murMurHash(eachIdx, seeds[iSeed]) % kSketches + sketchBeg;
				// should be safe: each thread accesses different vector elements
				if (sketches[newIdx] == emptyBinToken){
					sketches[newIdx] = sketches[eachIdx + sketchBeg];
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
	sketchTime = time2 - time1;
	return std::array<float, 2>{permTime.count(), sketchTime.count()};
}

int main() {
	BayesicSpace::RanDraw seedPRNG;
	const uint64_t seedInt  = seedPRNG.ranInt();
	BayesicSpace::RanDraw prng1(seedInt);
	BayesicSpace::RanDraw prng2(seedInt);
	constexpr size_t wordSizeBits    = 64;
	constexpr size_t wordSize        = 8;
	constexpr size_t byteSize        = 8;
	//constexpr size_t nIndividuals    = 1205;
	constexpr size_t nIndividuals    = 150;
	//constexpr size_t kSketches       = 100;
	constexpr size_t kSketches       = 20;
	constexpr size_t locusSize       = nIndividuals / byteSize + static_cast<size_t>( static_cast<bool>(nIndividuals % byteSize) );
	constexpr size_t sketchSize      = nIndividuals / kSketches;
	std::cout << sketchSize << "\n";
	constexpr uint16_t emptyBinToken = std::numeric_limits<uint16_t>::max();
	constexpr size_t ranBitVecSize   = nIndividuals / wordSizeBits + static_cast<size_t>( static_cast<bool> (nIndividuals % wordSizeBits) );
	std::vector<uint32_t> seeds1{static_cast<uint32_t>( prng1.ranInt() )};
	std::vector<uint32_t> seeds2{static_cast<uint32_t>( prng2.ranInt() )};
	std::vector<uint16_t> sketches1(kSketches, emptyBinToken);
	std::vector<uint16_t> sketches2(kSketches, emptyBinToken);
	std::vector<uint8_t> binLocus1;
	std::vector<uint8_t> binLocus2;
	std::vector<uint64_t> ranBits;
	for (size_t i = 0; i < ranBitVecSize; ++i){
		ranBits.push_back( seedPRNG.ranInt() );
	}
	size_t iByte = 0;
	binLocus1.resize(locusSize);
	for (const auto eachWord : ranBits){
		for (size_t byteInWord = 0; (byteInWord < wordSize) && (iByte < locusSize); ++byteInWord){
			binLocus1[iByte] = static_cast<uint8_t>( eachWord >> (byteInWord * byteSize) ) & (static_cast<uint8_t>(1) << byteInWord);
			++iByte;
		}
	}
	binLocus2 = binLocus1;
	for (auto blIt = binLocus1.rbegin(); blIt != binLocus1.rend(); ++blIt){
		std::cout << std::bitset<byteSize>(*blIt);
	}
	std::cout << "\n";
	//const size_t remainder = (nIndividuals % byteSize ? byteSize - nIndividuals % byteSize : 0);
	//binLocus.back() &= std::numeric_limits<uint8_t>::max() >> remainder;
	std::vector<size_t> ranIntsUp{seedPRNG.fyIndexesUp(nIndividuals)};
	const std::array<float, 2> res1 = locusOPHold(0, nIndividuals, locusSize, kSketches, sketchSize, ranIntsUp, seeds1, prng1, binLocus1, sketches1);
	const std::array<float, 2> res2 = locusOPH(0, nIndividuals, locusSize, kSketches, sketchSize, ranIntsUp, seeds2, prng2, binLocus2, sketches2);
	for (auto blIt = binLocus1.rbegin(); blIt != binLocus1.rend(); ++blIt){
		std::cout << std::bitset<byteSize>(*blIt);
	}
	std::cout << "\n";
	for (const auto eachSketch1 : sketches1){
		std::cout << eachSketch1 << " ";
	}
	std::cout << "\n";
	for (const auto eachSketch2 : sketches2){
		std::cout << eachSketch2 << " ";
	}
	std::cout << "\n";
	std::cout << res1[0] << "\t" << res1[1] << "\t" << res2[0] << "\t" << res2[1] << "\n";
}
